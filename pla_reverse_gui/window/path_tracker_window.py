"""Path tracker window"""
import numpy as np
from numba_pokemon_prngs.data.encounter import EncounterAreaLA
from numba_pokemon_prngs.data import NATURES_EN
from numba_pokemon_prngs.enums import LATime, LAWeather, LAArea
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection

# pylint: disable=no-name-in-module
from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QButtonGroup,
    QDialog,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from qtpy.QtGui import QIcon, QColor
from qtpy.QtCore import Qt, QSize, QTimer   # added QSize, QTimer

from pathlib import Path
from functools import partial

# pylint: enable=no-name-in-module
from ..util import calc_effort_level, get_personal_index, get_name_en, path_to_string, AREA_WEATHERS
from ..pla_reverse_main.pla_reverse.size import calc_display_size

TIME_DAYTIME = 100
TIME_ANYTIME = 101

class PathTableWidget(QTableWidget):
    """QTableWidget for spawner paths"""

    COLUMNS = (
        ("Caught", 60),
        ("Time", 80),
        ("Weather", 80),
        ("Advances", 90),
        ("Path", 140),
        ("Species", 120),
        ("Shiny", 60),
        ("Alpha", 60),
        ("Nature", 80),
        ("Effort Levels", 100),
        ("Gender", 70),
        ("Height", 80),
        ("Weight", 80),
    )

    def __init__(self):
        super().__init__()

        self.setColumnCount(13)
        self.setHorizontalHeaderLabels([column[0] for column in self.COLUMNS])
        for i, (_, width) in enumerate(self.COLUMNS):
            self.setColumnWidth(i, width)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.verticalHeader().setVisible(False)

        self.setMinimumHeight(500)
        self.setMinimumWidth(900)


class PathTrackerWindow(QDialog):
    def __init__(self,
                 parent,
                 encounter_table,
                 second_wave_encounter_table,
                 seed,
                 pre_path,
                 path,
                 count_values,
                 max_spawn_count,
                 weather,
                 time,
                 species_info,
                 initial_spawns=0,
                 area=None,
                 allow_other_starts=False):
        super().__init__(parent)

        # Store parameters (as before)
        self.encounter_table = encounter_table
        self.second_wave_encounter_table = second_wave_encounter_table
        self.seed = seed
        self.pre_path = pre_path
        self.path = path
        self.count_values = count_values
        self.max_spawn_count = max_spawn_count
        self.species_info = species_info
        self.initial_spawns = initial_spawns
        self.area = area
        self.allow_other_starts = allow_other_starts
        self.current_weather = weather
        self.current_time = time

        self.setWindowTitle("Path Tracker " + path_to_string(path))
        self.main_layout = QVBoxLayout(self)

        # Create table early (so it exists when signals fire)
        self.path_table = PathTableWidget()
        self._initializing = True

        # Build top widgets (time and weather)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        # Time selection
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time:"))
        self.btn_daytime = QPushButton()
        self.btn_daytime.setIcon(QIcon(self.get_icon_path("time_daytime.png")))
        self.btn_daytime.setIconSize(QSize(32, 32))
        self.btn_daytime.setToolTip("Daytime (Dawn, Day, Dusk)")
        self.btn_daytime.setCheckable(True)
        self.btn_daytime.toggled.connect(lambda checked: self.change_time(LATime.DAWN.value) if checked else None)

        self.btn_night = QPushButton()
        self.btn_night.setIcon(QIcon(self.get_icon_path("time_night.png")))
        self.btn_night.setIconSize(QSize(32, 32))
        self.btn_night.setToolTip("Night")
        self.btn_night.setCheckable(True)
        self.btn_night.toggled.connect(lambda checked: self.change_time(LATime.NIGHT.value) if checked else None)

        self.time_group = QButtonGroup(self)
        self.time_group.setExclusive(True)
        self.time_group.addButton(self.btn_daytime)
        self.time_group.addButton(self.btn_night)

        time_layout.addWidget(self.btn_daytime)
        time_layout.addWidget(self.btn_night)
        time_layout.addStretch()

        # Weather selection
        weather_layout = QHBoxLayout()
        weather_layout.addWidget(QLabel("Weather:"))
        self.weather_buttons = []
        self.weather_group = QButtonGroup(self)
        self.weather_group.setExclusive(True)

        if self.area is not None and self.area in AREA_WEATHERS:
            for weather in AREA_WEATHERS[self.area]:
                btn = QPushButton()
                icon = self.get_weather_icon(weather.value)
                if icon:
                    btn.setIcon(icon)
                    btn.setIconSize(QSize(32, 32))
                btn.setToolTip(weather.name.title())
                btn.setCheckable(True)
                # Set initial checked without emitting signal
                if weather.value == self.current_weather:
                    btn.blockSignals(True)
                    btn.setChecked(True)
                    btn.blockSignals(False)
                btn.toggled.connect(lambda checked, w=weather: self.change_weather(w.value) if checked else None)
                weather_layout.addWidget(btn)
                self.weather_buttons.append(btn)
                self.weather_group.addButton(btn)
        else:
            weather_layout.addWidget(QLabel("(unknown area)"))
        weather_layout.addStretch()

        top_layout.addLayout(time_layout)
        top_layout.addLayout(weather_layout)

        # Set initial time button checked (without triggering recalc)
        if self.current_time in (LATime.DAWN.value, LATime.DAY.value, LATime.DUSK.value, TIME_DAYTIME):
            self.btn_daytime.blockSignals(True)
            self.btn_daytime.setChecked(True)
            self.btn_daytime.blockSignals(False)
        elif self.current_time == LATime.NIGHT.value:
            self.btn_night.blockSignals(True)
            self.btn_night.setChecked(True)
            self.btn_night.blockSignals(False)

        # Path display label
        self.path_display_label = QLabel()
        self.path_display_label.setTextFormat(Qt.RichText)
        self.path_display_label.setWordWrap(True)
        self.update_path_display(path, 0)

        # Control buttons
        self.btn_reset = QPushButton("Reset (Z)")
        self.btn_reset.setShortcut(Qt.Key_Z)
        self.btn_reset.clicked.connect(self.on_reset)

        self.btn_undo = QPushButton("Undo (R)")
        self.btn_undo.setShortcut(Qt.Key_R)
        self.btn_undo.clicked.connect(self.on_undo)

        self.btn_done = QPushButton("Done (E)")
        self.btn_done.setShortcut(Qt.Key_E)
        self.btn_done.clicked.connect(self.on_done)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_reset)
        button_layout.addWidget(self.btn_undo)
        button_layout.addWidget(self.btn_done)
        button_layout.addStretch()

        control_row = QWidget()
        control_layout = QHBoxLayout(control_row)
        control_layout.addWidget(self.path_display_label, stretch=1)
        control_layout.addLayout(button_layout)

        # Add everything to main layout
        self.main_layout.addWidget(top_widget)
        self.main_layout.addWidget(control_row)
        self.main_layout.addWidget(self.path_table)

        self.resize(sum(c[1] for c in self.path_table.COLUMNS), self.height())

        # Finish initialization and run simulation
        self._initializing = False
        self.run_simulation()

    # ----------------------------------------------------------------------
    # Simulation logic (extracted from __init__)
    # ----------------------------------------------------------------------
    def run_simulation(self):
        """Clear and rebuild the table using current weather/time."""
        self.path_table.setRowCount(0)
        self.row_states = []

        current_encounter_table = self.encounter_table
        group_rng = Xoroshiro128PlusRejection(self.seed)
        ghost_count = 3

        count_vals = self.count_values
        if count_vals[0] != -1:
            # Variable multispawner
            full_sequence = (self.initial_spawns,) + self.path
            count_vals = (self.initial_spawns,) + self.pre_path + self.count_values
        else:
            full_sequence = self.pre_path + self.path

        pre_len = len(self.pre_path)
        index_start_modifier = pre_len if not self.allow_other_starts else 0

        # Keep track of the last advance value
        last_advance = None

        # Enumerate
        for advance, spawn_count in enumerate(full_sequence):
            is_ghost = False

            # Special values
            if spawn_count == 255:
                spawn_count = 4
                current_encounter_table = self.second_wave_encounter_table
            elif spawn_count > 20:
                continue
            elif spawn_count > 10:
                ghost_count -= spawn_count - 10
                spawn_count = 3 - ghost_count
                is_ghost = True

            # Build displayed path
            if advance < pre_len:
                current_path = ()
            else:
                conditional = len(self.pre_path) - 1 if len(self.pre_path) > 1 else 1
                current_path = full_sequence[1 : advance + conditional]

            # Variable multi logic
            if count_vals[0] != -1:
                current_spawn_count = count_vals[advance]
                count_before_spawns = current_spawn_count - spawn_count
                next_number_of_spawns = count_vals[advance + 1]
                generated = max(0, next_number_of_spawns - count_before_spawns)
                spawn_count = generated

            # Generate Pokémon
            for _ in range(spawn_count):
                generator_seed = np.uint64(group_rng.next())
                generator_rng = Xoroshiro128PlusRejection(generator_seed)
                group_rng.next()
                if is_ghost:
                    continue

                slot = current_encounter_table.calc_slot(
                    generator_rng.next() / 2**64,
                    np.int64(self.current_time),
                    np.int64(self.current_weather)
                )
                gender_ratio, shiny_rolls, _ = self.species_info[(slot.species, slot.form)]
                fixed_rng = Xoroshiro128PlusRejection(np.uint64(generator_rng.next()))
                fixed_rng.next_rand(0xFFFFFFFF)  # encryption constant
                sidtid = fixed_rng.next_rand(0xFFFFFFFF)
                shiny = 0
                for _ in range(shiny_rolls):
                    pid = fixed_rng.next_rand(0xFFFFFFFF)
                    xor = (
                        (pid >> 16)
                        ^ (sidtid >> 16)
                        ^ (pid & 0xFFFF)
                        ^ (sidtid & 0xFFFF)
                    )
                    shiny = 2 if xor == 0 else 1 if xor < 16 else 0
                    if shiny:
                        break

                effort_levels = np.zeros(6, np.uint8)
                for _ in range(slot.guaranteed_ivs):
                    idx = fixed_rng.next_rand(6)
                    while effort_levels[idx] != 0:
                        idx = fixed_rng.next_rand(6)
                    effort_levels[idx] = 3
                for i in range(6):
                    if effort_levels[i] == 0:
                        effort_levels[i] = calc_effort_level(fixed_rng.next_rand(32))

                fixed_rng.next_rand(2)  # ability
                gender = 0 if gender_ratio == 0 else 1 if gender_ratio == 254 else 2
                if 1 <= gender_ratio < 254:
                    gender = (fixed_rng.next_rand(253) + 1) < gender_ratio
                nature = fixed_rng.next_rand(25)
                if slot.is_alpha:
                    height = weight = 255
                else:
                    height = fixed_rng.next_rand(0x81) + fixed_rng.next_rand(0x80)
                    weight = fixed_rng.next_rand(0x81) + fixed_rng.next_rand(0x80)

                personal_index = get_personal_index(slot.species, slot.form)
                disp_metric = calc_display_size(personal_index, height, weight, imperial=False)
                disp_imperial = calc_display_size(personal_index, height, weight, imperial=True)

                row_i = self.path_table.rowCount()
                self.path_table.insertRow(row_i)
                # Determine advance value to display
                display_adv = advance - index_start_modifier

                # Create Caught button
                caught_btn = QPushButton()
                caught_btn.setFixedSize(40, 25)
                # We'll set text later
                self.path_table.setCellWidget(row_i, 0, caught_btn)

                # Time and Weather items (will be set later)
                time_item = QTableWidgetItem()
                weather_item = QTableWidgetItem()
                self.path_table.setItem(row_i, 1, time_item)
                self.path_table.setItem(row_i, 2, weather_item)

                # Store row state
                row_state = {
                    'advance': display_adv,
                    'locked': False,
                    'caught_state': None,   # None, '1', '2', '3', 'check'
                    'stored_time': self.current_time,
                    'stored_weather': self.current_weather,
                    'row_i': row_i,
                    'button': caught_btn,
                    'time_item': time_item,
                    'weather_item': weather_item,
                }
                self.row_states.append(row_state)

                # Connect button click
                caught_btn.clicked.connect(partial(self.on_caught_clicked, row_i))

                row_data = (
                    str(advance - index_start_modifier),
                    path_to_string(current_path),
                    get_name_en(slot.species, slot.form, slot.is_alpha),
                    "Square" if shiny == 2 else "Star" if shiny else "No",
                    "Yes" if slot.is_alpha else "No",
                    NATURES_EN[nature],
                    "/".join(str(x) for x in effort_levels),
                    "♂" if gender == 0 else "♀" if gender == 1 else "○",
                    f"{disp_metric[0]:.02f} m | {disp_imperial[0][0]:.00f}'{disp_imperial[0][1]:.00f}\" ({height})",
                    f"{disp_metric[1]:.02f} kg | {disp_imperial[1]:.01f} lbs ({weight})",
                )
                for j, value in enumerate(row_data, start=3):
                    item = QTableWidgetItem(value)
                    self.path_table.setItem(row_i, j, item)

            if self.row_states:
                max_adv = max((s['advance'] for s in self.row_states), default=None)
                for state in self.row_states:
                    if state['advance'] < 0:
                        state['locked'] = True
                        state['caught_state'] = 'check'
                        # stored_time/weather already = initial values (since they were set during generation)
                    elif state['advance'] == max_adv:
                        state['locked'] = True
                        state['caught_state'] = None   # not caught, but time/weather frozen to original
                        # stored_time/weather already = initial values
                    else:
                        state['locked'] = False
                        state['caught_state'] = None

                    # Update button text and time/weather icons for all rows
                    self.update_row_states()

            if spawn_count != 0:
                group_rng.re_init(np.uint64(group_rng.next()))

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def get_icon_path(self, icon_name: str) -> str:
        current_dir = Path(__file__).parent
        package_root = current_dir.parent
        icon_path = package_root / "Resources" / "Icons" / icon_name
        return str(icon_path)
    
    def get_time_icon(self, time_val: int):
        if time_val == TIME_DAYTIME:
            path = self.get_icon_path("time_daytime.png")
            return QIcon(path) if path else None
        if time_val == TIME_ANYTIME:
            path = self.get_icon_path("time_anytime.png")
            return QIcon(path) if path else None
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

    def get_weather_icon(self, weather_val: int):
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

    def update_path_display(self, path_tuple: tuple[int], current_index: int):
        parts = [str(step) for step in path_tuple]
        if 0 <= current_index < len(parts):
            parts[current_index] = f"<big><b>{parts[current_index]}</b></big>"
        path_str = " → ".join(parts)
        self.path_display_label.setText(f"Path: {path_str}")

    def on_time_clicked(self, time_val, checked):
        self.change_time(time_val)

    def on_weather_clicked(self, weather_val, checked):
        self.change_weather(weather_val)
    
    def change_time(self, time_val):
        self.current_time = time_val
        if not self._initializing:
            self.recalculate()

    def change_weather(self, weather_val):
        self.current_weather = weather_val
        if not self._initializing:
            self.recalculate()

    def update_row_states(self):
        for state in self.row_states:
            # Update button text
            btn = state['button']
            if state['caught_state'] == 'check':
                btn.setText("✓")
            elif state['caught_state'] in ('1','2','3'):
                btn.setText(f"({state['caught_state']})")
            else:
                btn.setText("")

            # Determine time and weather to display
            if state['locked']:
                time_val = state['stored_time']
                weather_val = state['stored_weather']
            else:
                time_val = self.current_time
                weather_val = self.current_weather

            # Set icons
            time_icon = self.get_time_icon(time_val)
            weather_icon = self.get_weather_icon(weather_val)
            state['time_item'].setIcon(time_icon if time_icon else QIcon())
            state['weather_item'].setIcon(weather_icon if weather_icon else QIcon())
            # Optionally set tooltips
            state['time_item'].setToolTip(f"Time: {time_val}")
            state['weather_item'].setToolTip(f"Weather: {weather_val}")

            # Set row background color if locked (caught)
            if state['locked'] and state['caught_state'] is not None:
                for col in range(self.path_table.columnCount()):
                    item = self.path_table.item(state['row_i'], col)
                    if item:
                        item.setBackground(QColor(35, 55, 75))
            else:
                # Reset background
                for col in range(self.path_table.columnCount()):
                    item = self.path_table.item(state['row_i'], col)
                    #if item:
                        #item.setBackground(QColor(255, 255, 255))

    def on_caught_clicked(self, row_i):
        state = self.row_states[row_i]
        if state['locked']:
            return  # already locked, cannot change
        # Cycle caught_state: None -> '1' -> '2' -> '3' -> 'check' -> None
        cycle = [None, '1', '2', '3', 'check']
        current = state['caught_state']
        idx = cycle.index(current) if current in cycle else 0
        next_state = cycle[(idx + 1) % len(cycle)]
        state['caught_state'] = next_state
        if next_state is not None:
            # Lock the row and store current weather/time
            state['locked'] = True
            state['stored_time'] = self.current_time
            state['stored_weather'] = self.current_weather
        else:
            # Unlock? According to spec, once caught it stays caught. But we allow toggling off.
            state['locked'] = False
        self.update_row_states()

    def recalculate(self):
        self.run_simulation()

    def flash_button(self, button):
        original_style = button.styleSheet()
        button.setStyleSheet("background-color: lightblue;")
        QTimer.singleShot(200, lambda: button.setStyleSheet(original_style))

    def on_reset(self):
        self.flash_button(self.btn_reset)
        # TODO: actual reset logic

    def on_undo(self):
        self.flash_button(self.btn_undo)

    def on_done(self):
        self.flash_button(self.btn_done)