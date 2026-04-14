"""Spawner generator window"""

import time
import numpy as np
import numba
import sys
from numba.typed import List as TypedList
from numba_pokemon_prngs.data.encounter import (
    ENCOUNTER_TABLE_NAMES_LA,
    SPAWNER_NAMES_LA,
    EncounterAreaLA,
)
from numba_pokemon_prngs.data import NATURES_EN, ABILITIES_EN
from numba_pokemon_prngs.data.fbs.encounter_la import PlacementSpawner8a
from numba_pokemon_prngs.enums import LAWeather, LATime, LAArea

# pylint: disable=no-name-in-module
from numba.typed import Dict as TypedDict
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidgetItem,
)
from qtpy.QtGui import QRegularExpressionValidator, QIcon, QPainter, QPixmap, QColor
from qtpy import QtCore
from qtpy.QtCore import QThread, Signal, Qt, QSettings, QSize

from pathlib import Path

# pylint: enable=no-name-in-module

from .result_table_widget import ResultTableWidget
from ..util import get_name_en, get_personal_info, get_personal_index, path_to_string, AREA_WEATHERS
from .checkable_combobox_widget import CheckableComboBox
from .range_widget import RangeWidget
from ..generator import generate_standard, generate_mass_outbreak, generate_variable
from ..pla_reverse_main.pla_reverse.size import calc_display_size
from .eta_progress_bar import ETAProgressBar

TIME_DAYTIME = 100
TIME_ANYTIME = 101

class NoIndentDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Remove any decoration role (icon/checkmark)
        option.decorationSize = QSize(0, 0)
        option.features &= ~QStyleOptionViewItem.HasDecoration
        # Draw the text only
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        # Return size without decoration column
        text = index.data(Qt.DisplayRole)
        metrics = option.fontMetrics
        width = metrics.horizontalAdvance(text) + 8  # small padding
        height = metrics.height() + 4
        return QSize(width, height)

def compute_result_count(max_spawn_count: int, max_path_length: int, allow_other_starts: bool) -> int:
    """Calculate the total amount of results to be generated for a given spawner, max path length and if the user allows other starting paths"""
    if max_spawn_count == 1:
        return max_path_length
    if max_spawn_count == 4:
        initial_value = 1
    else:
        ## In the case of 2 or 3 max spawn counts
        initial_value = (max_spawn_count - 1) + allow_other_starts * 1
    return initial_value * (1 - max_spawn_count**max_path_length) // (1 - max_spawn_count)

def compute_result_count_variable(spawn_counts: list[int], allow_other_starts: bool, initial_spawns: int) -> int:
    """
    Count the number of internal states for a variable spawner.
    spawn_counts: list of target spawn counts for each user step.
    initial_spawns: 1, 2, or 3 (the first KO count and initial field size).
    """
    from collections import defaultdict

    # dp maps (last_ko, cur_spawn) -> number of states at current depth
    dp = defaultdict(int)
    dp[(initial_spawns, initial_spawns)] = 1   # root state
    total_internal = 1

    for depth, target in enumerate(spawn_counts):
        new_dp = defaultdict(int)
        for (last_ko, cur_spawn), count in dp.items():
            before = cur_spawn - last_ko
            generated = max(0, target - before)
            after = before + generated
            for kos in range(after + 1):
                new_dp[(kos, after)] += count
                if depth + 1 < len(spawn_counts):
                    total_internal += count
        dp = new_dp

    if allow_other_starts:
        if initial_spawns == 1:
            multiplier = 4
        elif initial_spawns == 2:
            multiplier = 3
        else:  # initial_spawns == 3
            multiplier = 4
        total_internal *= multiplier

    return total_internal

def labled_widget(label: str, widget_constructor: QWidget, *args, **kwargs) -> tuple[QWidget, QWidget]:
    outer = QWidget()
    layout = QHBoxLayout(outer)
    layout.setSpacing(2)
    layout.addWidget(QLabel(label))
    widget = widget_constructor(*args, **kwargs)
    layout.addWidget(widget)
    return widget, outer


class GeneratorWindow(QDialog):
    """Spawner generator window"""

    def __init__(
        self,
        parent: QWidget,
        spawner: PlacementSpawner8a,
        encounter_table: EncounterAreaLA,
        second_wave_encounter_table: EncounterAreaLA,
        area: LAArea,
    ) -> None:
        super().__init__(parent)
        self.area = area
        self.generator_update_thread = None
        self.spawner = spawner
        self.encounter_table = encounter_table
        self.second_wave_encounter_table = second_wave_encounter_table
        self.has_second_wave = self.second_wave_encounter_table is not None
        self.is_mmo = spawner.encounter_table_id != self.encounter_table.table_id
        is_variable = spawner.min_spawn_count != spawner.max_spawn_count
        self.basculin_gender = {
            0xFD9CA9CA1D5681CB: 0,  # M
            0xFD999DCA1D543790: 1,  # F
        }.get(spawner.encounter_table_id, None)
        self.setWindowTitle(
            "Generator "
            f"{SPAWNER_NAMES_LA.get(np.uint64(spawner.spawner_id), '')} - "
            f"{ENCOUNTER_TABLE_NAMES_LA.get(np.uint64(self.encounter_table.table_id), '')}"
        )
        self.main_layout = QVBoxLayout(self)
        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout(self.top_widget)
        self.settings_widget = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_widget)
        self.settings_layout.addWidget(QLabel("Seed:"))
        self.seed_input = QLineEdit()
        self.seed_base_combobox = QComboBox()
        self.seed_base_combobox.addItem("Hexadecimal", 16)
        self.seed_base_combobox.addItem("Decimal", 10)

        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.toggle_settings_display_button = QPushButton("▼ Hide Settings")
        self.toggle_settings_display_button.setCheckable(True)
        self.toggle_settings_display_button.setChecked(True)
        self.toggle_settings_display_button.clicked.connect(self.hide_show_settings)
        self.header_layout.addWidget(self.toggle_settings_display_button)

        self.id_to_weather = {}      # id -> set of weather values
        self.id_to_time = {}         # id -> set of time values

        def seed_base_changed(index: int) -> None:
            is_hex = index == 0
            previous_text = self.seed_input.text()
            seed_value = int(previous_text, 10 if is_hex else 16) if previous_text else 0
            if seed_value >= (1 << 64):
                seed_value = 0
            self.seed_input.setValidator(
                QRegularExpressionValidator(
                    QtCore.QRegularExpression(
                        "[0-9a-fA-F]{0,16}" if is_hex else "[0-9]{0,20}"
                    )
                )
            )
            self.seed_input.setText(
                (f"{seed_value:X}" if is_hex else f"{seed_value}") if seed_value else ""
            )
        self.seed_base_combobox.currentIndexChanged.connect(seed_base_changed)
        seed_base_changed(0)
        self.settings_layout.addWidget(self.seed_base_combobox)
        self.settings_layout.addWidget(self.seed_input)
        settings_label = QLabel("Settings:")
        self.settings_layout.addWidget(settings_label)
        self.weather_combobox, weather_widget = labled_widget("Weather:", QComboBox)
        self.weather_combobox: QComboBox
        if not self.spawner.is_mass_outbreak:
            self.weather_combobox.addItem("All Weathers", None)
        for weather in LAWeather:
            if weather != LAWeather.NONE:
                self.weather_combobox.addItem(weather.name.title(), weather)
        self.time_combobox, time_widget = labled_widget("Time:", QComboBox)
        self.time_combobox: QComboBox
        if not self.spawner.is_mass_outbreak:
            self.time_combobox.addItem("All Times", None)
        for time in LATime:
            self.time_combobox.addItem(time.name.title(), time)
        self._time_equiv_cache = {}
        settings_label.setVisible(not self.spawner.is_mass_outbreak)
        weather_widget.setVisible(not self.spawner.is_mass_outbreak)
        time_widget.setVisible(not self.spawner.is_mass_outbreak)
        self.settings_layout.addWidget(weather_widget)
        self.settings_layout.addWidget(time_widget)
        spawn_count_label = QLabel("Spawn Count:")
        spawn_count_label.setVisible(bool(self.spawner.is_mass_outbreak))
        self.settings_layout.addWidget(spawn_count_label, 2, )
        if self.has_second_wave:
            self.first_wave_spawn_count, first_wave_spawn_count_widget = labled_widget("First Wave:", QSpinBox, minimum=8, maximum=10)
        else:
            self.first_wave_spawn_count, first_wave_spawn_count_widget = labled_widget("First Wave:", QSpinBox, minimum=10, maximum=15)
        first_wave_spawn_count_widget.setVisible(bool(self.spawner.is_mass_outbreak))
        self.second_wave_spawn_count, second_wave_spawn_count_widget = labled_widget("Second Wave:", QSpinBox, minimum=6, maximum=8)
        second_wave_spawn_count_widget.setVisible(self.has_second_wave)
        self.settings_layout.addWidget(first_wave_spawn_count_widget)
        self.settings_layout.addWidget(second_wave_spawn_count_widget)
        advance_range_label = QLabel("Advance Range:")
        self.settings_layout.addWidget(advance_range_label)
        self.advance_range = RangeWidget(0, 20 if self.spawner.min_spawn_count > 1 else 9999)
        self.advance_range.max_entry.setMaximum(99999999)
        advance_range_label.setVisible(not (self.spawner.is_mass_outbreak or is_variable))
        self.advance_range.setVisible(not (self.spawner.is_mass_outbreak or is_variable))
        self.settings_layout.addWidget(self.advance_range)
        self.settings_layout.addWidget(QLabel("Shiny Rolls:"))
        self.shiny_roll_entries = []

        self.added_species = []
        self.unique_slots = set()
        self.shiny_rolls_comboboxes = {}
        for slot in self.encounter_table.slots.view(np.recarray):
            self.unique_slots.add((slot.species, slot.form))
            if slot.species in self.added_species:
                continue
            self.added_species.append(slot.species)
            species_name = get_name_en(slot.species, None)

            shiny_rolls_combobox, shiny_rolls_outer = labled_widget(species_name, QComboBox)
            for item in (
                ("Base Research", 1),
                ("Research Level 10", 2),
                ("Perfect Research", 4),
                ("Shiny Charm + Research Level 10", 5),
                ("Shiny Charm + Perfect Research", 7),
            ):
                shiny_rolls_combobox.addItem(*item)

            settings = QSettings("PLAReverseGUI", "Settings")
            shiny_charm = settings.value("shinyCharm", False, bool)
            research_level = settings.value(f"species/{slot.species}/researchLevel", 0, int)
            if research_level == 0:
                target_index = 3 if shiny_charm else 0
            elif research_level == 1:
                target_index = 3 if shiny_charm else 1
            else:
                target_index = 4 if shiny_charm else 2
            shiny_rolls_combobox.setCurrentIndex(target_index)

            self.settings_layout.addWidget(shiny_rolls_outer)
            self.shiny_rolls_comboboxes[
                slot.species
            ] = shiny_rolls_combobox
        self.allow_other_starts_checkbox = QCheckBox("Allow paths that do not start with catch-2")
        self.allow_other_starts_checkbox.setVisible(self.spawner.max_spawn_count > 1)
        self.settings_layout.addWidget(self.allow_other_starts_checkbox)
        
        if is_variable:
            initial_spawn_options = []
            
            if self.spawner.min_spawn_count == 1:
                initial_spawn_options.append("1->1")
            initial_spawn_options.append("2")
            if self.spawner.max_spawn_count == 3:
                initial_spawn_options.append("3")
            
            spawn_count_extended = QHBoxLayout()
            
            initial_spawn_layout = QVBoxLayout()
            self.initial_spawn_label = QLabel("Initial\nspawns:")
            initial_spawn_layout.addWidget(self.initial_spawn_label)
            self.initial_spawn_box = QComboBox()
            self.initial_spawn_box.addItems(initial_spawn_options)
            initial_spawn_layout.addWidget(self.initial_spawn_box)
            self.initial_spawn_box.setItemDelegate(NoIndentDelegate())

            spawn_count_layout = QVBoxLayout()
            starting_path_label = QLabel("Spawn Count Values:")
            spawn_count_layout.addWidget(starting_path_label)
            self.starting_path_input = QLineEdit()
            spawn_count_layout.addWidget(self.starting_path_input)
            
            spawn_count_extended.addLayout(initial_spawn_layout)
            spawn_count_extended.addLayout(spawn_count_layout)
            self.settings_layout.addLayout(spawn_count_extended)
            
        else:
            starting_path_label = QLabel("Starting Path:")
            self.settings_layout.addWidget(starting_path_label)
            self.starting_path_input = QLineEdit()
            self.settings_layout.addWidget(self.starting_path_input)
        
        starting_path_label.setVisible(self.spawner.max_spawn_count > 1 and not self.spawner.is_mass_outbreak)
        # TODO: regex validation
        # self.starting_path_input.setValidator(
        #     QRegularExpressionValidator(QtCore.QRegularExpression(""))
        # )
        self.starting_path_input.setVisible(self.spawner.max_spawn_count > 1 and not self.spawner.is_mass_outbreak)

        self.filter_widget = QWidget()
        self.filter_layout = QVBoxLayout(self.filter_widget)

        self.species_filter, species_widget = labled_widget("Species Filter:", CheckableComboBox)
        self.species_filter: CheckableComboBox
        for species_form in self.unique_slots:
            self.species_filter.add_checked_item(get_name_en(*species_form), species_form)

        self.gender_filter, gender_widget = labled_widget("Gender Filter:", CheckableComboBox)
        self.gender_filter: CheckableComboBox
        self.gender_filter.add_checked_item("Male", 0)
        self.gender_filter.add_checked_item("Female", 1)
        self.nature_filter, nature_widget = labled_widget("Nature:", CheckableComboBox)
        self.nature_filter: CheckableComboBox
        for i, nature in enumerate(NATURES_EN):
            self.nature_filter.add_checked_item(nature, i)
        self.shiny_filter, shiny_widget = labled_widget("Shiny Filter:", QComboBox)
        self.shiny_filter: QComboBox
        self.shiny_filter.addItem("Any", None)
        self.shiny_filter.addItem("Star", 1)
        self.shiny_filter.addItem("Square", 2)
        self.shiny_filter.addItem("Star/Square", 1 | 2)
        self.alpha_filter = QCheckBox("Alpha Only")
        self.shortest_path_filter = QCheckBox("Only Shortest Path")
        self.shortest_path_filter.setChecked(True)
        self.chain_results_filter = QCheckBox("Only Chain Results")
        self.shortest_path_filter.clicked.connect(self.shortest_path_or_chain_results)
        self.chain_results_filter.clicked.connect(self.shortest_path_or_chain_results)

        self.size_filter, size_widget = labled_widget("Height/Scale:", CheckableComboBox)
        self.size_filter: CheckableComboBox
        self.size_filter.add_checked_item("XXXS (0)", 0)
        self.size_filter.add_checked_item("XXXL (255)", 255)

        self.filter_layout.addWidget(species_widget)
        self.filter_layout.addWidget(gender_widget)
        self.filter_layout.addWidget(nature_widget)
        self.filter_layout.addWidget(shiny_widget)
        self.filter_layout.addWidget(size_widget)
        self.filter_layout.addWidget(self.alpha_filter)
        self.filter_layout.addWidget(self.shortest_path_filter)
        self.filter_layout.addWidget(self.chain_results_filter)

        self.iv_filter_widget = QWidget()
        self.iv_filter_layout = QVBoxLayout(self.iv_filter_widget)
        self.iv_filters = (
            RangeWidget(0, 31, "HP:"),
            RangeWidget(0, 31, "Atk:"),
            RangeWidget(0, 31, "Def:"),
            RangeWidget(0, 31, "SpA:"),
            RangeWidget(0, 31, "SpD:"),
            RangeWidget(0, 31, "Spe:"),
        )
        for iv_filter in self.iv_filters:
            self.iv_filter_layout.addWidget(iv_filter)

        self.top_layout.addWidget(self.settings_widget)
        self.top_layout.addWidget(self.iv_filter_widget)
        self.top_layout.addWidget(self.filter_widget)

        self.progress_bar = ETAProgressBar()
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate)

        self.result_table = ResultTableWidget()
        self.result_table.area = area
        self.result_table.allow_other_starts = self.allow_other_starts_checkbox.isChecked()
        self.result_table.parent_window = self
        self.main_layout.addWidget(self.header_widget)
        self.main_layout.addWidget(self.top_widget)
        self.main_layout.addWidget(self.generate_button)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.result_table)

        settings = QSettings("PLAReverseGUI", "Settings")
        if settings.value("alwaysSearchShiny", False, bool):
            shiny_type = settings.value("shinyType", 2, int)
            if shiny_type == 0:
                self.shiny_filter.setCurrentIndex(1)
            elif shiny_type == 1:
                self.shiny_filter.setCurrentIndex(2)
            else:
                self.shiny_filter.setCurrentIndex(3)
        if settings.value("alwaysSearchAlpha", False, bool):
            self.alpha_filter.setChecked(True)

        self.resize(
            sum(column[1] for column in self.result_table.COLUMNS),
            self.height(),
        )
        self.row_weather_sets = {}
        self.row_time_sets = {}

    def hide_show_settings(self):
        visible = self.toggle_settings_display_button.isChecked()
        self.top_widget.setVisible(visible)
        self.toggle_settings_display_button.setText("▼ Hide Settings" if visible else "▶ Display Settings")

    def get_distinct_combos(self, area: EncounterAreaLA, time_filter=None, weather_filter=None, allowed_weathers=None):
        times = list(LATime)
        
        if time_filter is None and not self.spawner.is_mass_outbreak:  # All Times
            # Use only DAWN and NIGHT as representatives
            times = [LATime.DAWN, LATime.NIGHT]
        else:
            times = list(LATime)
        
        if weather_filter is not None:
            weathers = [weather_filter]
        elif allowed_weathers is not None:
            weathers = allowed_weathers
        else:
            weathers = list(LAWeather)

        base_probs = area.slots['base_probability']
        time_mult = area.slots['time_multipliers']
        weather_mult = area.slots['weather_multipliers']
        t_ref = LATime.DUSK
        t_idx = t_ref.value

        vector_by_weather = {}
        for w in weathers:
            w_idx = w.value
            probs = base_probs * time_mult[:, t_idx] * weather_mult[:, w_idx]
            total = np.sum(probs)
            norm = probs / total if total > 0 else probs
            vector_by_weather[w.value] = tuple(np.round(norm, decimals=6))

        groups = {}
        for w_val, vec in vector_by_weather.items():
            groups.setdefault(vec, []).append(w_val)

        rep_to_weathers = {}
        weather_to_rep = {}
        for vec, wlist in groups.items():
            rep = min(wlist)
            rep_to_weathers[rep] = wlist
            for w in wlist:
                weather_to_rep[w] = rep

        combos = []
        for t in times:
            if time_filter is not None and t != time_filter:
                continue
            reps_seen = set()
            for w_val in vector_by_weather.keys():
                rep = weather_to_rep[w_val]
                if rep not in reps_seen:
                    reps_seen.add(rep)
                    combos.append((t, rep))

        return combos, rep_to_weathers

    def _get_time_category_map(self, weather_val):
        if weather_val in self._time_equiv_cache:
            return self._time_equiv_cache[weather_val]

        base_probs = self.encounter_table.slots['base_probability']
        time_mult = self.encounter_table.slots['time_multipliers']
        weather_mult = self.encounter_table.slots['weather_multipliers']
        w_idx = weather_val

        vectors = {}
        for t in LATime:
            probs = base_probs * time_mult[:, t.value] * weather_mult[:, w_idx]
            total = np.sum(probs)
            norm = probs / total if total > 0 else probs
            vectors[t] = tuple(np.round(norm, decimals=6))

        groups = {}
        for t, vec in vectors.items():
            groups.setdefault(vec, []).append(t)

        category_map = {}
        for vec, times in groups.items():
            time_set = set(times)
            if time_set == {LATime.DAWN, LATime.DAY, LATime.DUSK}:
                cat = 'daytime'
            elif time_set == {LATime.NIGHT}:
                cat = 'night'
            elif time_set == set(LATime):
                cat = 'anytime'
            else:
                cat = None
            for t in times:
                category_map[t.value] = cat

        self._time_equiv_cache[weather_val] = category_map
        return category_map

    def get_time_set(self, result_id):
        return self.id_to_time.get(result_id, set())

    def set_time_set(self, result_id, time_set):
        self.id_to_time[result_id] = time_set
        row = self.find_row_by_id(result_id)
        if row != -1:
            time_item = self.result_table.item(row, 3)
            time_item.setData(Qt.UserRole, sorted(list(time_set)))
            time_item.setToolTip(self.format_time_tooltip(sorted(list(time_set))))
            self.result_table.viewport().update()

    def get_time_icon(self, time_val, weather_val=None):
        if time_val == TIME_DAYTIME:
            path = self.get_icon_path('time_daytime.png')
            return QIcon(path) if path else None
        if time_val == TIME_ANYTIME:
            path = self.get_icon_path('time_anytime.png')
            return QIcon(path) if path else None
        if weather_val is not None:
            cat_map = self._get_time_category_map(weather_val)
            cat = cat_map.get(time_val)
            if cat == 'daytime':
                path = self.get_icon_path('time_daytime.png')
                if path:
                    return QIcon(path)
            elif cat == 'night':
                path = self.get_icon_path('time_night.png')
                if path:
                    return QIcon(path)
            elif cat == 'anytime':
                path = self.get_icon_path('time_anytime.png')
                if path:
                    return QIcon(path)
        try:
            time_name = LATime(time_val).name.lower()
        except ValueError:
            return None
        icon_map = {
            'dawn': 'time_dawn.png',
            'day': 'time_day.png',
            'dusk': 'time_dusk.png',
            'night': 'time_night.png',
        }
        filename = icon_map.get(time_name)
        if filename:
            path = self.get_icon_path(filename)
            return QIcon(path)
        return None

    def format_time_tooltip(self, time_vals):
        if not time_vals:
            return ""
        if len(time_vals) == 1:
            v = time_vals[0]
            if v == TIME_DAYTIME:
                return "Dawn, Day, Dusk"
            elif v == TIME_ANYTIME:
                return "Dawn, Day, Dusk, Night"
        names = []
        for t in sorted(time_vals):
            try:
                names.append(LATime(t).name.title())
            except ValueError:
                names.append(f"Time {t}")
        return ", ".join(names)

    def get_weather_set(self, result_id):
        return self.id_to_weather.get(result_id, set())

    def set_weather_set(self, result_id, weather_set):
        self.id_to_weather[result_id] = weather_set
        row = self.find_row_by_id(result_id)
        if row != -1:
            weather_item = self.result_table.item(row, 2)
            weather_item.setData(Qt.UserRole, sorted(list(weather_set)))
            weather_item.setToolTip(self.format_weather_tooltip(sorted(list(weather_set))))
            self.result_table.viewport().update()

    def get_weather_icon(self, weather_val):
        try:
            weather_name = LAWeather(weather_val).name.lower()
        except ValueError:
            return None
        icon_map = {
            'sunny': 'weather_sunny.png',
            'cloudy': 'weather_cloudy.png',
            'rain': 'weather_rain.png',
            'snow': 'weather_snow.png',
            'drought': 'weather_drought.png',
            'fog': 'weather_fog.png',
            'rainstorm': 'weather_rainstorm.png',
            'snowstorm': 'weather_snowstorm.png',
            'none': 'weather_any.png',
        }
        filename = icon_map.get(weather_name)
        if filename:
            path = self.get_icon_path(filename)
            return QIcon(path)
        return None

    def format_weather_tooltip(self, weather_vals):
        if not weather_vals:
            return ""
        if len(weather_vals) == 1 and weather_vals[0] == 0:
            return "Any Weather"
        names = []
        for w in sorted(weather_vals):
            if w == 0:
                names.append("Any Weather")
            else:
                try:
                    names.append(LAWeather(w).name.title())
                except ValueError:
                    names.append(f"Weather {w}")
        return ", ".join(names)

    def get_icon_path(self, icon_name: str) -> str:
        current_dir = Path(__file__).parent
        package_root = current_dir.parent
        icon_path = package_root / "Resources" / "Icons" / icon_name
        return str(icon_path)

    def find_row_by_id(self, target_id):
        for row in range(self.result_table.rowCount()):
            item = self.result_table.item(row, 19)  # hidden ID column
            if item and item.data(Qt.UserRole) == target_id:
                return row
        return -1

    def generate(self) -> None:
        self.result_table.setRowCount(0)
        seed = int(seed_str, self.seed_base_combobox.currentData()) if (seed_str := self.seed_input.text()) else 0
        if seed >= (1 << 64):
            self.seed_input.setText("")
            return
        seed = np.uint64(seed)
        extra_shiny_rolls = 0
        if self.spawner.is_mass_outbreak:
            extra_shiny_rolls = 25
            if self.is_mmo:
                extra_shiny_rolls = 12
        starting_path = tuple(int(x) for x in self.starting_path_input.text().split("->") if x)
        if len(starting_path) == 0:
            starting_path = (-1,)
        advance_range = self.advance_range.get_range()
        species_info = TypedDict.empty(key_type=numba.typeof((0,0)), value_type=numba.typeof((0,0,False)))

        filtered_species = self.species_filter.get_checked_values()
        filtered_genders = self.gender_filter.get_checked_values()
        filtered_natures = self.nature_filter.get_checked_values()
        filtered_sizes = self.size_filter.get_checked_values()
        # Parse initial spawns for variable spawners – use first character as integer
        if hasattr(self, 'initial_spawn_box'):
            selected = self.initial_spawn_box.currentText()
            # "1->1" becomes 1, "2" becomes 2, "3" becomes 3
            initial_spawns = int(selected[0])
        else:
            initial_spawns = self.spawner.max_spawn_count
            
        shiny_filter = self.shiny_filter.currentData() or 15
        alpha_filter = self.alpha_filter.checkState() == QtCore.Qt.Checked
        shortest_path_filter = self.shortest_path_filter.checkState() == QtCore.Qt.Checked
        chain_results_filter = self.chain_results_filter.isChecked()
        iv_filters = tuple((iv_range.start, iv_range.stop-1) for iv_range in (f.get_range() for f in self.iv_filters))

        for species, form in self.unique_slots:
            pi = get_personal_info(species, form)
            species_info[(species, form)] = (
                self.basculin_gender if (species, form)==(550,2) and self.basculin_gender is not None else pi.gender_ratio,
                self.shiny_rolls_comboboxes[species].currentData() + extra_shiny_rolls,
                len(filtered_species)==0 or (species, form) in filtered_species,
            )

        weather_data = self.weather_combobox.currentData()
        time_data = self.time_combobox.currentData()

        if self.spawner.is_mass_outbreak:
            combos = [(weather_data, time_data)]
            rep_to_weathers = {}      # empty dict – will not be used in mass outbreak
            full_map_weathers = None
        else:
            time_filter = None if time_data is None else time_data
            weather_filter = None if weather_data is None else weather_data
            if weather_data is None:
                base_area = LAArea(self.area & 0xFF)
                allowed = [LAWeather.NONE] + AREA_WEATHERS.get(base_area, [])
                combos, rep_to_weathers = self.get_distinct_combos(
                    self.encounter_table, time_filter, None, allowed_weathers=allowed
                )
                full_map_weathers = set(w.value for w in AREA_WEATHERS.get(base_area, []))
                other_reps = set(rep for rep in rep_to_weathers if rep != 0)
                other_weathers = set()
                for rep in other_reps:
                    other_weathers.update(rep_to_weathers[rep])
                none_weathers = full_map_weathers - other_weathers
                rep_to_weathers[0] = list(none_weathers)
            else:
                combos, rep_to_weathers = self.get_distinct_combos(
                    self.encounter_table, time_filter, weather_filter
                )
                full_map_weathers = None
                
        # For time expansion (hardcoded because DAWN/DAY/DUSK are always same in PLA)
        rep_to_times = {
            LATime.DAWN.value: [LATime.DAWN.value, LATime.DAY.value, LATime.DUSK.value],
            LATime.NIGHT.value: [LATime.NIGHT.value],
        }
        # If the user selected a specific time, we need a mapping for that single time
        if time_data is not None:
            # For a specific time, the representative is that time itself
            rep_to_times = {time_data.value: [time_data.value]}

        if self.spawner.is_mass_outbreak or self.spawner.min_spawn_count != self.spawner.max_spawn_count:
            per_combo = compute_result_count_variable(starting_path, self.allow_other_starts_checkbox.isChecked(), initial_spawns)
        else:
            per_combo = compute_result_count(self.spawner.max_spawn_count, advance_range.stop, self.allow_other_starts_checkbox.isChecked())

        per_combo_progress = [per_combo] * len(combos)
        total_progress = per_combo * len(combos)
        self.progress_bar.setMaximum(total_progress)

        # Build base_args
        if self.spawner.is_mass_outbreak:
            base_args = (
                seed,
                self.allow_other_starts_checkbox.isChecked(),
                self.first_wave_spawn_count.value(),
                self.second_wave_spawn_count.value() if self.has_second_wave else 0,
                self.encounter_table,
                self.second_wave_encounter_table or self.second_wave_encounter_table,
                species_info,
                filtered_genders,
                filtered_natures,
                filtered_sizes,
                shiny_filter,
                alpha_filter,
                iv_filters,
            )
        elif self.spawner.min_spawn_count != self.spawner.max_spawn_count:
            base_args = (
                seed,
                self.allow_other_starts_checkbox.isChecked(),
                initial_spawns,
                starting_path,
                self.spawner.max_spawn_count,
                self.encounter_table,
                species_info,
                filtered_genders,
                filtered_natures,
                filtered_sizes,
                shiny_filter,
                alpha_filter,
                iv_filters,
            )
        else:
            base_args = (
                seed,
                self.allow_other_starts_checkbox.isChecked(),
                initial_spawns,
                starting_path,
                advance_range.start,
                advance_range.stop,
                self.spawner.max_spawn_count,
                self.encounter_table,
                species_info,
                filtered_genders,
                filtered_natures,
                filtered_sizes,
                shiny_filter,
                alpha_filter,
                iv_filters,
            )

        self.generator_update_thread = GeneratorUpdateThread(
            self,
            self.spawner.is_mass_outbreak,
            self.spawner.min_spawn_count != self.spawner.max_spawn_count,
            shortest_path_filter,
            chain_results_filter,
            combos,
            per_combo_progress,
            rep_to_weathers,
            full_map_weathers,
            rep_to_times,
            *base_args
        )
        self.generator_update_thread.progress.connect(self.progress_bar.setValue)

        def cleanup_generate():
            if self.generator_update_thread is not None:
                self.generator_update_thread.requestInterruption()
            self.generate_button.setText("Generate")
            self.generate_button.clicked.disconnect(cleanup_generate)
            self.generate_button.clicked.connect(self.generate)
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setMaximum(1)
                self.progress_bar.setValue(1)

        self.generate_button.setText("Cancel")
        self.generate_button.clicked.disconnect(self.generate)
        self.generate_button.clicked.connect(cleanup_generate)
        self.generator_update_thread.finished.connect(cleanup_generate)
        self.generator_update_thread.start()

        self.result_table.min_spawn_count = self.spawner.min_spawn_count
        self.result_table.max_spawn_count = self.spawner.max_spawn_count
        self.result_table.encounter_table = self.encounter_table
        self.result_table.second_wave_encounter_table = self.second_wave_encounter_table
        self.result_table.seed = seed
        self.result_table.weather = self.weather_combobox.currentData()
        self.result_table.time = self.time_combobox.currentData()
        self.result_table.species_info = species_info
        self.result_table.spawn_counts = starting_path
        self.result_table.initial_spawns = initial_spawns
        self.result_table.allow_other_starts = self.allow_other_starts_checkbox.isChecked()
        self.result_table.first_wave_count = self.first_wave_spawn_count.value() if self.spawner.is_mass_outbreak else 0
        
        # Remove the time and weather items if it is an MO/MMO
        self.result_table.setColumnHidden(2, self.spawner.max_spawn_count > 3)
        self.result_table.setColumnHidden(3, self.spawner.max_spawn_count > 3)

    def add_result(self, row: tuple, group_tuple: tuple, result_id: int):
        (
            advance,
            path,
            (species, form, is_alpha),
            _encryption_constant,
            _pid,
            ivs,
            ability,
            gender,
            nature,
            shiny,
            height,
            weight,
            weather_val,
            time_val,
        ) = row

        pi = get_personal_info(species, form)
        pidx = get_personal_index(species, form)
        disp_metric = calc_display_size(pidx, height, weight, imperial=False)
        disp_imperial = calc_display_size(pidx, height, weight, imperial=True)

        row_i = self.result_table.rowCount()
        self.result_table.insertRow(row_i)

        advance_item = QTableWidgetItem(str(advance))
        path_str = path_to_string(path) if self.spawner.max_spawn_count != 1 else "N/A"
        path_item = QTableWidgetItem(path_str)

        weather_item = QTableWidgetItem()
        weather_icon = self.get_weather_icon(weather_val)
        if weather_icon:
            weather_item.setIcon(weather_icon)
        weather_item.setText(LAWeather(weather_val).name.title() if weather_val < len(LAWeather) else "Unknown")
        weather_item.setData(Qt.UserRole, [weather_val])
        weather_item.setToolTip(self.format_weather_tooltip([weather_val]))

        time_item = QTableWidgetItem()
        time_icon = self.get_time_icon(time_val, weather_val)
        if time_icon:
            time_item.setIcon(time_icon)
        time_item.setText(LATime(time_val).name.title() if time_val < len(LATime) else "Unknown")
        time_item.setData(Qt.UserRole, [time_val])
        time_item.setToolTip(self.format_time_tooltip([time_val]))

        # Wurmple evolution
        if species == 265:
            first16 = (_encryption_constant >> 16) & 0xFFFF
            extension = "-Cascoon" if (first16 % 10) > 4 else "-Silcoon"
        else:
            extension = ""
        species_name = get_name_en(species, form, is_alpha) + extension
        species_item = QTableWidgetItem(species_name)

        shiny_str = "Square" if shiny == 2 else "Star" if shiny else "No"
        shiny_item = QTableWidgetItem(shiny_str)
        alpha_str = "Yes" if is_alpha else "No"
        alpha_item = QTableWidgetItem(alpha_str)
        nature_item = QTableWidgetItem(NATURES_EN[nature])
        ability_name = ABILITIES_EN[pi.ability_2 if ability else pi.ability_1]
        ability_item = QTableWidgetItem(ability_name)
        iv_items = [QTableWidgetItem(str(iv)) for iv in ivs]
        gender_str = "♂" if gender == 0 else "♀" if gender == 1 else "○"
        gender_item = QTableWidgetItem(gender_str)
        height_str = f"{disp_metric[0]:.02f} m | {disp_imperial[0][0]:.00f}'{disp_imperial[0][1]:.00f}\" ({height})"
        height_item = QTableWidgetItem(height_str)
        weight_str = f"{disp_metric[1]:.02f} kg | {disp_imperial[1]:.01f} lbs ({weight})"
        weight_item = QTableWidgetItem(weight_str)

        # Group string for display
        group_str = '.'.join(f"{part:03d}" for part in group_tuple)
        group_item = QTableWidgetItem(group_str)

        # Hidden ID
        id_item = QTableWidgetItem()
        id_item.setData(Qt.UserRole, result_id)

        # Generate sort key
        path_str_sort = ''.join(str(step) for step in path)
        sort_key = f"{advance:02d}" + path_str_sort
        sort_item = QTableWidgetItem(sort_key)

        items = [
            advance_item, path_item, weather_item, time_item, species_item,
            shiny_item, alpha_item, nature_item, ability_item,
            *iv_items, gender_item, height_item, weight_item,
            group_item, id_item, sort_item   # group at col 18, ID at col 19, SortKey at 20
        ]
        for j, item in enumerate(items):
            self.result_table.setItem(row_i, j, item)

        # Apply background color to entire row based on root group parity
        if self.chain_results_filter.isChecked():
            root_parity = group_tuple[0] % 2
            if root_parity == 1:   # color odd root groups
                bg_color = QColor(35, 55, 75)
                for col in range(self.result_table.columnCount()):
                    item = self.result_table.item(row_i, col)
                    if item:
                        item.setBackground(bg_color)

        self.id_to_weather[result_id] = {weather_val}
        self.id_to_time[result_id] = {time_val}
        self.row_weather_sets[row_i] = {weather_val}
        self.row_time_sets[row_i] = {time_val}
        if self.chain_results_filter.isChecked():
            self.result_table.model().sort(18, Qt.AscendingOrder)   # sort by group ID
        else:
            self.result_table.model().sort(20, Qt.AscendingOrder)   # sort by SortKey

        return row_i

    def update_result(self, result_id: int, weather_val: int, time_val: int):
        row = self.find_row_by_id(result_id)
        if row == -1:
            return

        wset = self.id_to_weather.get(result_id, set())
        wset.add(weather_val)
        self.id_to_weather[result_id] = wset
        witem = self.result_table.item(row, 2)
        witem.setData(Qt.UserRole, sorted(list(wset)))
        witem.setToolTip(self.format_weather_tooltip(sorted(list(wset))))

        tset = self.id_to_time.get(result_id, set())
        tset.add(time_val)
        self.id_to_time[result_id] = tset
        titem = self.result_table.item(row, 3)
        titem.setData(Qt.UserRole, sorted(list(tset)))
        titem.setToolTip(self.format_time_tooltip(sorted(list(tset))))

        self.result_table.viewport().update()

    def closeEvent(self, event):
        if self.generator_update_thread is not None:
            self.generator_update_thread.requestInterruption()
            self.generator_update_thread.wait()
        event.accept()

    def shortest_path_or_chain_results(self):
        if self.sender().isChecked():
            if self.sender() == self.shortest_path_filter:
                self.chain_results_filter.setChecked(False)
            else:
                self.shortest_path_filter.setChecked(False)


class GeneratorUpdateThread(QThread):
    finished = Signal()
    progress = Signal(int)
    new_result = Signal(tuple)

    def __init__(self, parent_window: GeneratorWindow, is_mass_outbreak: bool, is_variable: bool,
                 shortest_path_only: bool, chain_results: bool, combos, per_combo_progress,
                 rep_to_weathers, full_map_weathers, rep_to_times, *args) -> None:
        super().__init__()
        self.parent_window = parent_window
        self.is_mass_outbreak = is_mass_outbreak
        self.is_variable = is_variable
        self.shortest_path_only = shortest_path_only
        self.chain_results = chain_results
        self.combos = combos
        self.per_combo_progress = per_combo_progress
        self.rep_to_weathers = rep_to_weathers
        self.full_map_weathers = full_map_weathers
        self.rep_to_times = rep_to_times
        self.key_to_id = {}
        self.next_id = 0
        self.args = args

    def run(self):
        cumulative = 0
        # Data structures keyed by hidden ID (used for both added and non‑added)
        path_to_ids = {}          # path tuple -> list of hidden IDs (all)
        id_to_row = {}            # hidden ID -> modified row tuple (with first weather)
        added_ids = set()         # hidden IDs that have been added to table
        id_to_group = {}          # hidden ID -> group tuple (only for added IDs)
        group_child_counter = {}  # parent group tuple -> next child index
        next_root_id = 0

        for idx, (time_val, weather_val) in enumerate(self.combos):
            if self.isInterruptionRequested():
                break

            parent_data = np.zeros(2, np.uint64)
            full_args = self.args + (weather_val, time_val, parent_data)

            generator_thread = GeneratorThread(
                self.is_mass_outbreak,
                self.is_variable,
                *full_args
            )
            generator_thread.start()

            while generator_thread.isRunning():
                if self.isInterruptionRequested():
                    parent_data[1] = 1
                    break
                self.progress.emit(cumulative + parent_data[0])
                time.sleep(0.1)

            generator_thread.wait()

            for row in generator_thread.results:
                species, form, is_alpha = row[2]
                ec = row[3]
                pid = row[4]
                path_tuple = tuple(row[1])
                if self.shortest_path_only:
                    key = (species, form, is_alpha, ec, pid)
                else:
                    key = (species, form, is_alpha, ec, pid, path_tuple)

                weather_val = row[12]
                weather_represented = self.rep_to_weathers.get(weather_val, [weather_val])
                
                time_val = row[13]
                time_represented = self.rep_to_times.get(time_val, [time_val])

                # Determine which weathers to actually add to the row's set
                if self.full_map_weathers is not None:
                    rep_without_zero = set(weather_represented) - {0}
                    if rep_without_zero == self.full_map_weathers:
                        weathers_to_add = [0]
                    else:
                        weathers_to_add = [w for w in weather_represented if w != 0]
                else:
                    weathers_to_add = [weather_val]

                if not weathers_to_add:
                    continue

                # Find the longest parent path prefix that already has any hidden ID
                parent_id = None
                if self.chain_results:
                    parent_prefixes = []
                    for i in range(1, len(path_tuple)):
                        prefix = path_tuple[:i]
                        if prefix in path_to_ids:
                            parent_prefixes.append(prefix)
                    if parent_prefixes:
                        longest_parent = max(parent_prefixes, key=len)
                        parent_id = path_to_ids[longest_parent][0]  # any ID, added or not

                # Weather/time aggregation via hidden ID
                if key in self.key_to_id:
                    result_id = self.key_to_id[key]
                    first_w = weathers_to_add[0]
                    first_t = time_represented[0]
                    for w in weathers_to_add:
                        self.parent_window.update_result(result_id, w, first_t)
                    for t in time_represented:
                        self.parent_window.update_result(result_id, first_w, t)
                else:
                    result_id = self.next_id
                    self.key_to_id[key] = result_id
                    self.next_id += 1
                    path_to_ids.setdefault(path_tuple, []).append(result_id)
                    first_w = weathers_to_add[0]
                    first_t = time_represented[0]
                    modified_row = list(row)
                    modified_row[12] = first_w
                    modified_row[13] = first_t
                    modified_row = tuple(modified_row)
                    id_to_row[result_id] = modified_row

                    # Determine if we should add this result to the table now
                    add_now = not self.chain_results or (self.chain_results and parent_id is not None)

                    if add_now:
                        # Determine group tuple
                        if parent_id is not None and parent_id in added_ids:
                            # Parent already in table – use its group and append child
                            parent_group = id_to_group[parent_id]
                            child_num = group_child_counter.get(parent_group, 1)
                            group_tuple = parent_group + (child_num,)
                            group_child_counter[parent_group] = child_num + 1
                        elif parent_id is not None:
                            # Parent exists but not yet added – add it first
                            parent_row = id_to_row[parent_id]
                            parent_group = (next_root_id,)
                            next_root_id += 1
                            self.parent_window.add_result(parent_row, parent_group, parent_id)
                            added_ids.add(parent_id)
                            id_to_group[parent_id] = parent_group
                            # Now child gets first child of that new root
                            child_num = 1
                            group_tuple = parent_group + (child_num,)
                            group_child_counter[parent_group] = child_num + 1
                        else:
                            # No parent – new root
                            group_tuple = (next_root_id,)
                            next_root_id += 1

                        # Add the current result
                        self.parent_window.add_result(modified_row, group_tuple, result_id)
                        added_ids.add(result_id)
                        id_to_group[result_id] = group_tuple

                    # Add any remaining weathers (updates)
                    for w in weathers_to_add[1:]:
                        self.parent_window.update_result(result_id, w, row[13])
                    # Add remaining times (using first_w)
                    for t in time_represented[1:]:
                        self.parent_window.update_result(result_id, first_w, t)

            cumulative += self.per_combo_progress[idx]
            self.progress.emit(cumulative)

        # After all combos, collapse weather/time sets (unchanged)
        if self.full_map_weathers is not None:
            for result_id in list(self.key_to_id.values()):
                current_set = self.parent_window.get_weather_set(result_id)
                if current_set and (current_set - {0}) == self.full_map_weathers:
                    self.parent_window.set_weather_set(result_id, {0})

        for result_id in list(self.key_to_id.values()):
            current_times = self.parent_window.get_time_set(result_id)
            if current_times == {LATime.DAWN.value, LATime.DAY.value, LATime.DUSK.value}:
                self.parent_window.set_time_set(result_id, {TIME_DAYTIME})
            elif current_times == {t.value for t in LATime}:
                self.parent_window.set_time_set(result_id, {TIME_ANYTIME})

        self.finished.emit()


class GeneratorThread(QThread):
    finished = Signal()

    def __init__(self, is_outbreak: bool, is_variable: bool, *args) -> None:
        super().__init__()
        self.is_outbreak = is_outbreak
        self.is_variable = is_variable
        self.args = args
        self.results = TypedList.empty_list(
            item_type=numba.typeof(
                (
                    0,
                    [np.uint8(0)],
                    (np.uint16(0), np.uint8(0), np.bool_(0)),
                    np.uint32(0),
                    np.uint32(0),
                    np.zeros(6, np.uint8),
                    np.uint8(0),
                    np.uint8(0),
                    np.uint8(0),
                    np.uint8(0),
                    np.uint8(0),
                    np.uint8(0),
                    np.uint8(0),
                    np.uint8(0),
                )
            )
        )

    def run(self) -> None:
        if self.is_outbreak:
            generate_mass_outbreak(*self.args, self.results)
        elif self.is_variable:
            generate_variable(*self.args, self.results)
        else:
            generate_standard(*self.args, self.results)
        self.finished.emit()
