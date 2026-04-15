import numpy as np
from numba_pokemon_prngs.data.encounter import EncounterAreaLA
from numba_pokemon_prngs.data import NATURES_EN
from numba_pokemon_prngs.enums import LATime, LAWeather, LAArea
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection

from qtpy.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QButtonGroup, QDialog, QLabel, QPushButton,
    QSizePolicy, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QWidget,
)
from qtpy.QtGui import QIcon, QColor, QPen
from qtpy.QtCore import Qt, QSize, QTimer
from pathlib import Path
from functools import partial

from ..util import calc_effort_level, get_personal_index, get_name_en, path_to_string, AREA_WEATHERS
from ..pla_reverse_main.pla_reverse.size import calc_display_size

import numbers

TIME_DAYTIME = 100
TIME_ANYTIME = 101

class PathTableWidget(QTableWidget):
    COLUMNS = (
        ("0xAdvance", 0),   # hidden, not used
        ("StepInAdv", 0),   # hidden, not used
        ("Locked", 0),      # hidden, not used
        ("Caught", 70),
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
        self.setColumnCount(16)
        self.setHorizontalHeaderLabels([c[0] for c in self.COLUMNS])
        for i, (_, w) in enumerate(self.COLUMNS):
            self.setColumnWidth(i, w)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.verticalHeader().setVisible(False)
        # Optional: hide columns 0-2
        for i in range(3):
            self.setColumnHidden(i, True)
        delegate = IconDelegate(self, icon_size=24)
        self.setItemDelegateForColumn(4, delegate)  # Time column (index 4)
        self.setItemDelegateForColumn(5, delegate)  # Weather column (index 5)
        self.setMinimumHeight(300)
        self.setMinimumWidth(750)

class IconDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, icon_size=28):
        super().__init__(parent)
        self.icon_size = icon_size
    def paint(self, painter, option, index):
        icon = index.data(Qt.DecorationRole)
        if icon and isinstance(icon, QIcon):
            rect = option.rect
            pixmap = icon.pixmap(self.icon_size, self.icon_size)
            x = rect.x() + (rect.width() - pixmap.width()) // 2 + 10
            y = rect.y() + (rect.height() - pixmap.height()) // 2 + 12
            painter.drawPixmap(x, y, pixmap)
        else:
            super().paint(painter, option, index)

class PathTrackerWindow(QDialog):
    def __init__(self, parent, encounter_table, second_wave_encounter_table, seed,
                 pre_path, path, count_values, max_spawn_count, weather, time,
                 species_info, spawn_counts, initial_spawns=0, area=None, allow_other_starts=False,
                 first_wave_count=0):
        super().__init__(parent)
        self.encounter_table = encounter_table
        self.second_wave_encounter_table = second_wave_encounter_table
        self.seed = seed
        self.pre_path = pre_path
        self.path = path
        self.count_values = count_values
        self.max_spawn_count = max_spawn_count
        self.species_info = species_info
        self.spawn_counts = spawn_counts
        self.initial_spawns = initial_spawns
        self.area = area
        self.allow_other_starts = allow_other_starts
        self.current_weather = weather
        self.current_time = time

        self.last_advance_time = time
        self.last_advance_weather = weather

        # Normalize sentinel time values to a valid LATime value for simulation
        if self.current_time in (TIME_DAYTIME, TIME_ANYTIME):
            self.current_time = LATime.DAWN.value
        # If "any weather" (NONE or not in area list), default to first area weather
        if self.area is not None and self.area in AREA_WEATHERS:
            area_weathers = AREA_WEATHERS[self.area]
            if area_weathers and not any(w.value == self.current_weather for w in area_weathers):
                self.current_weather = area_weathers[0].value

        self.first_wave_count = first_wave_count
        self.is_mo = (self.max_spawn_count == 4 and self.spawn_counts[0] == -1)

        # MO/MMO: step indexing starts at 0 (path already begins at first tracked batch)
        if allow_other_starts or self.is_mo:
            self.step_correction = 0
        else:
            self.step_correction = len(pre_path)

        self.current_step = self.step_correction

        # persistent storage: row_index -> dict
        self.stored_rows = {}                       # If the user did not select to allow other starts, stores all rows that are considered to be caught pokemon (initial catches)
        self.last_advance_conditions = {}           # The last advance mons (the resulting mons from the path) stored time and weather to help the user be reminded of them
        self.current_ko_count = 0                   # The current KO count of mons at any point in time at that step
        self.caught_index = 0                       # Order of the pokemon caught
        self.initialize_mon_number = 0              # Index of the initial mons that need to be stored
        self.initializing = True                    # Simple bool to stop any function after initializing
        self.col_idx = 0                            # Column index for storing the time and weather of the last mons
        self.initial_number_to_store = 0            # Number of initial pokemon to store and lock (0 if the user allows other starts, the number of mons in the pre-path if not)
        self.is_flashing = False                    # Fail-safe in case the path is blinking due to doing a wrong step
        self.update_caught_index(type='reset')
        self.insert_separator = [False, -1]         # Position to add the separator if there is a second wave

        self.setWindowTitle("Path Tracker " + path_to_string(path))
        self.main_layout = QVBoxLayout(self)
        self.path_table = PathTableWidget()
        self._initializing = True

        # Hide time and weather columns when it is a MO/MMO
        self.path_table.setColumnHidden(4, max_spawn_count > 3)
        self.path_table.setColumnHidden(5, max_spawn_count > 3)

        # ---------- build top widgets (time/weather) ----------
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time:"))
        self.btn_daytime = QPushButton()
        self.btn_daytime.setIcon(QIcon(self.get_icon_path("time_daytime.png")))
        self.btn_daytime.setIconSize(QSize(24,24))
        self.btn_daytime.setToolTip("Daytime (Dawn, Day, Dusk)")
        self.btn_daytime.setCheckable(True)
        self.btn_daytime.toggled.connect(lambda checked: self.change_time(LATime.DAWN.value) if checked else None)
        self.btn_night = QPushButton()
        self.btn_night.setIcon(QIcon(self.get_icon_path("time_night.png")))
        self.btn_night.setIconSize(QSize(24,24))
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
                    btn.setIconSize(QSize(32,32))
                btn.setToolTip(weather.name.title())
                btn.setCheckable(True)
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

        # initial time button check
        if self.current_time in (LATime.DAWN.value, LATime.DAY.value, LATime.DUSK.value, TIME_DAYTIME):
            self.btn_daytime.blockSignals(True)
            self.btn_daytime.setChecked(True)
            self.btn_daytime.blockSignals(False)
        elif self.current_time == LATime.NIGHT.value:
            self.btn_night.blockSignals(True)
            self.btn_night.setChecked(True)
            self.btn_night.blockSignals(False)

        # path display label
        self.path_display_label = QLabel()
        self.path_display_label.setTextFormat(Qt.RichText)
        self.path_display_label.setWordWrap(True)
        self.update_path_display(path, 0)

        # control buttons
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

        self.main_layout.addWidget(top_widget)
        # MO/MMO spawners have no weather/time variation — hide the selector row
        if self.is_mo:
            top_widget.hide()
        self.main_layout.addWidget(control_row)
        self.main_layout.addWidget(self.path_table)
        self.resize(sum(c[1] for c in self.path_table.COLUMNS), self.height())

        self._initializing = False
        self.run_simulation(scroll_back_up=True)

    # ----------------------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------------------
    def run_simulation(self, scroll_back_up):
        """(Re-)Generate the list of mons for a specific path, weather and time of day"""
        if not scroll_back_up:
            scroll_pos = self.path_table.verticalScrollBar().value()
        else:
            scroll_pos = 0
        self.path_table.setRowCount(0)

        current_encounter_table = self.encounter_table
        group_rng = Xoroshiro128PlusRejection(self.seed)
        ghost_count = 3

        count_vals = self.spawn_counts
        if count_vals[0] != -1:
            # For variable multispawners, we need to append the initial spawns
            # to spawn the initial mons that we got the seed from (whether the first 1 from 1->1 or 2->)
            # The pre-path for variable multispawners already includes a pre-path, so it ends up
            # being initial spawns (1, 2 or 3) + path (pre-path + path after the initial catches)
            if self.allow_other_starts:
                if self.initial_spawns == 1:
                    count_vals = (self.initial_spawns, self.initial_spawns,) + self.pre_path + self.spawn_counts
                    full_sequence = (self.initial_spawns, self.initial_spawns) + self.path
                else:
                    count_vals = (self.initial_spawns,) + self.pre_path + self.spawn_counts
                    full_sequence = (self.initial_spawns,) + self.path
                current_spawn_count = self.initial_spawns
            else:
                full_sequence = (self.initial_spawns,) + self.path

                # For the count values, we also need to add the initial spawns before the pre-path and the
                # count values so as to run an 'empty run' to spawn the first 1, 2 or 3 mons
                count_vals = (self.initial_spawns,) + self.pre_path + self.spawn_counts
                current_spawn_count = self.initial_spawns
        else:
            # For regular multispawners or MO/MMO mons, the pre-path is the initial mons that spawn.
            if self.is_mo:
                # MO: prepend the initial 4-spawn as a visible batch so it appears in the table.
                # The RNG naturally starts at seed, so no skip needed — processing 4 pokemon here
                # then calling re_init afterwards is exactly equivalent to the old skip.
                full_sequence = (np.uint8(4),) + self.pre_path + self.path
            else:
                full_sequence = self.pre_path + self.path

        pre_len = len(self.pre_path)
        if self.is_mo:
            # effective_pre_len: advances 0..pre_len are all pre-path (initial spawn + pre_path batches).
            effective_pre_len = pre_len + 1
            if self.allow_other_starts:
                # Initial at display_adv = 0, interactive path starts at 1.
                index_start_modifier = 0
            else:
                # Initial at display_adv = -pre_len (e.g. -3), last pre_path spawn at 0,
                # interactive path starts at 1.
                index_start_modifier = pre_len
        else:
            index_start_modifier = pre_len if not self.allow_other_starts else 0
            effective_pre_len = pre_len

        for advance, spawn_count in enumerate(full_sequence):
            is_ghost = False
            # special values
            if spawn_count == 255:
                self.insert_separator[0] = True
                spawn_count = 4
                current_encounter_table = self.second_wave_encounter_table
            elif spawn_count > 20:
                continue
            elif spawn_count > 10:
                ghost_count -= spawn_count - 10
                spawn_count = 3 - ghost_count
                is_ghost = True

            # build displayed path
            if advance < effective_pre_len:
                current_path = ()
            elif self.is_mo:
                # Show interactive path progress: path elements processed so far
                interactive_step = advance - effective_pre_len
                current_path = self.path[:interactive_step + 1]
            else:
                conditional = len(self.pre_path) - 1 if len(self.pre_path) > 1 else 1
                current_path = full_sequence[1 : advance + conditional]

            # variable multi logic
            if count_vals[0] != -1:
                count_before_spawns = current_spawn_count - spawn_count
                spawn_count = max(0, count_vals[advance + 1] - count_before_spawns)
                current_spawn_count = count_before_spawns + spawn_count
                #cur_spawn = count_vals[advance]
                #before = cur_spawn - spawn_count
                #next_spawn = count_vals[advance + 1]
                #generated = max(0, next_spawn - before)
                #spawn_count = generated

            for idx_in_step in range(spawn_count):
                # ---- Pokémon generation (RNG, slot, etc.) ----
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
                fixed_rng.next_rand(0xFFFFFFFF)
                sidtid = fixed_rng.next_rand(0xFFFFFFFF)
                shiny = 0
                for _ in range(shiny_rolls):
                    pid = fixed_rng.next_rand(0xFFFFFFFF)
                    xor = (pid>>16) ^ (sidtid>>16) ^ (pid&0xFFFF) ^ (sidtid&0xFFFF)
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
                fixed_rng.next_rand(2)
                gender = 0 if gender_ratio == 0 else 1 if gender_ratio == 254 else 2
                if 1 <= gender_ratio < 254:
                    gender = (fixed_rng.next_rand(253)+1) < gender_ratio
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
                display_adv = advance - index_start_modifier

                # Caught button (col 3)
                caught_btn = QPushButton()
                caught_btn.setFixedSize(70,30)
                caught_btn.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        border: none;
                        color: palette(text);
                        text-align: center;
                        outline: none;
                        border-radius: 0px;
                    }
                    QPushButton:hover {
                        background: rgba(255, 255, 255, 20);
                    }
                    QPushButton:pressed {
                        background: rgba(255, 255, 255, 40);
                    }
                """)
                self.path_table.setCellWidget(row_i, 3, caught_btn)

                # Time (col4) and Weather (col5) items
                time_item = QTableWidgetItem()
                weather_item = QTableWidgetItem()
                advance_item = QTableWidgetItem()
                time_item.setText("")
                weather_item.setText("")
                advance_item.setText(str(advance))
                self.path_table.setItem(row_i, 4, time_item)
                self.path_table.setItem(row_i, 5, weather_item)
                self.path_table.setItem(row_i, 0, advance_item)

                # visible columns start at col6
                row_data = (
                    str(display_adv),
                    path_to_string(current_path),
                    get_name_en(slot.species, slot.form, slot.is_alpha),
                    "Square" if shiny==2 else "Star" if shiny else "No",
                    "Yes" if slot.is_alpha else "No",
                    NATURES_EN[nature],
                    "/".join(str(x) for x in effort_levels),
                    "♂" if gender==0 else "♀" if gender==1 else "○",
                    f"{disp_metric[0]:.02f} m | {disp_imperial[0][0]:.00f}'{disp_imperial[0][1]:.00f}\" ({height})",
                    f"{disp_metric[1]:.02f} kg | {disp_imperial[1]:.01f} lbs ({weight})",
                )
                if self.insert_separator[0]:
                    self.insert_separator = [False, row_i]
                for j, val in enumerate(row_data, start=6):
                    self.path_table.setItem(row_i, j, QTableWidgetItem(val))

                # store metadata
                # For MO default start: only specific rows are pre-caught.
                # Rule: from initial 4-spawn, assume player caught the LAST pokemon (idx == spawn_count-1).
                # Respawns at advances 1..effective_pre_len-2 are each caught.
                # The final respawn at advance effective_pre_len-1 stays in the field (not caught).
                if self.is_mo and not self.allow_other_starts:
                    if advance == 0:
                        is_pre_caught = (idx_in_step == spawn_count - 1)
                    elif advance == effective_pre_len - 1:
                        is_pre_caught = False   # last respawn stays in field
                    else:
                        is_pre_caught = True    # intermediate respawns are caught
                else:
                    is_pre_caught = True
                self.store_initial_rows(advance, is_pre_caught)

                # connect button
                caught_btn.clicked.connect(partial(self.on_caught_clicked, row_i))

            if spawn_count != 0:
                group_rng.re_init(np.uint64(group_rng.next()))

        # --- reapply stored rows (locked) ---
        for row_i, stored in self.stored_rows.items():
            if not stored.get('locked', False):
                continue
            if row_i >= self.path_table.rowCount():
                continue
            btn = self.path_table.cellWidget(row_i, 3)
            if btn:
                btn.setText(stored.get('button_text', ''))
            time_item = self.path_table.item(row_i, 4)
            weather_item = self.path_table.item(row_i, 5)
            if time_item:
                time_item.setData(Qt.DecorationRole, self.get_time_icon(stored['stored_time']))
            if weather_item:
                weather_item.setData(Qt.DecorationRole, self.get_weather_icon(stored['stored_weather']))
            col_vals = stored.get('col_values', [])
            for j, val in enumerate(col_vals, start=6):
                item = self.path_table.item(row_i, j)
                if item:
                    item.setText(val)
            for col in range(self.path_table.columnCount()):
                item = self.path_table.item(row_i, col)
                if item:
                    item.setBackground(QColor(35,55,75))

        delegate = DoubleBorderDelegate(self.path_table, separator_row=self.insert_separator[1])
        self.path_table.setItemDelegate(delegate)
        self.renumber_step_buttons()
        self.update_path_display(self.path, self.current_step)
        self.update_last_advance_weather()
        self.initializing = False
        self.initialize_mon_number = 0
        self.path_table.verticalScrollBar().setValue(scroll_pos)
        

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def update_caught_index(self, type):
        """Update the number corresponding to the order in which the pokemon was caught"""
        if type == 'reset':
            self.caught_index = 0
            if self.is_mo and not self.allow_other_starts:
                # Rows stored as pre-caught: last of initial 4 (1) + intermediate pre-path (pre_len-1)
                # = pre_len total.  The last respawn (advance == effective_pre_len-1) stays in field.
                self.initial_number_to_store = len(self.pre_path)
                self.caught_index = self.initial_number_to_store
            elif not self.allow_other_starts:
                for sc in self.pre_path:
                    self.caught_index += sc
                    self.initial_number_to_store += sc
        if type == 'undo':
            # For MO ghost/clear wave, path value encodes 11-14 or 255 rather than catch count.
            if self.is_mo:
                self.caught_index -= self._step_ko_necessary(self.current_step)
            else:
                self.caught_index -= self.path[self.current_step]

    def store_initial_rows(self, adv, is_pre_caught=True):
        """Function to store either in the locked rows either no pokemon if allowing other starts, or the number of pokemon from the pre-path"""
        if not self.initializing:
            return
        if not self.allow_other_starts:
            if self.is_mo:
                # For MO: only store rows that were actually pre-caught so that
                # uncaught rows (initial 4 minus the last, and the last respawn) remain
                # visible and are numbered by renumber_step_buttons.
                should_store = self.col_idx in [3, 4, 5]
            else:
                should_store = self.initialize_mon_number < self.initial_number_to_store
            if should_store:
                self.stored_rows[self.col_idx] = {
                        'locked': True,
                        'advance': self.current_step,
                        # Use count of already-stored rows so caught_number stays 0-based
                        # and below initial_number_to_store for undo/reset guard.
                        'caught_number': len(self.stored_rows),
                        'button_text': "✓",
                        'stored_time': self.current_time,
                        'stored_weather': self.current_weather,
                        'col_values': self.get_row_values(self.col_idx),
                    }
        if self.is_mo:
            # Last advance index in full_sequence = 1 (initial) + pre_len + len(path) - 1
            last_advance_number = len(self.pre_path) + len(self.path)
        else:
            last_advance_number = len(self.path)
            if self.initial_spawns == 1:
                last_advance_number += 1
        if adv == last_advance_number:
            self.last_advance_conditions[self.col_idx] = {
                'advance': last_advance_number,
                'stored_time': self.last_advance_time,
                'stored_weather': self.last_advance_weather,
            }
        self.col_idx += 1
        self.initialize_mon_number += 1

    def get_row_values(self, row_i):
        """Return list of text values for columns 6..15 (visible data)."""
        values = []
        for col in range(6, self.path_table.columnCount()):
            item = self.path_table.item(row_i, col)
            values.append(item.text() if item else "")
        return values

    def renumber_step_buttons(self):
        """(Re-)Number the pokemon that will spawn at this step, and assign shortcuts"""
        # Clear all button texts on rows that are not caught
        for row_i in range(self.path_table.rowCount()):
            if row_i not in self.stored_rows:
                btn = self.path_table.cellWidget(row_i, 3)
                if btn:
                    btn.setText("")
                    btn.setShortcut(0)

        number_of_buttons = 0
        if self.spawn_counts[0] == -1:
            if self.is_mo:
                # MO/MMO field size = min(4, total_remaining).
                # total_remaining = first_wave_count - all pokemon caught so far.
                # pre_path catches are already done; path[:current_step] are interactive catches done.
                if self.current_step < len(self.path):
                    step_path_val = self.path[self.current_step]
                    print()
                    print(f"1: Step path val: {step_path_val}")
                    print(f"Path: {self.path}")
                    if step_path_val > 10:
                        # Ghost or Clear Wave: use _mo_field_at_step which correctly handles
                        # ghost chains (field refills after normal steps, depletes through ghost chain).
                        number_of_buttons = self._mo_field_at_step(self.current_step)
                    else:
                        total_caught = sum(self.pre_path) + sum(self.path[:self.current_step])
                        number_of_buttons = min(4, self.first_wave_count - total_caught) if total_caught < 255 else 4
                        print()
                        print(f"Total caught: {total_caught}, number of buttons: {number_of_buttons}")
                else:
                    number_of_buttons = 0
            else:
                # Regular fixed-size multispawner: field always has max_spawn_count slots
                number_of_buttons = self.max_spawn_count
        else:
            # Variable multispawner: simulate actual field size at current step.
            # Rule: spawner will never remove extra pokemon if the dictated count
            # drops below the actual count, but will spawn more if dictated count rises.
            # For initial_spawns=1, the two initial catches [1,1] are one atomic unit:
            # spawn_counts[0] applies after path[1], not path[0]. Use spawn_counts_offset=1.
            actual = self.initial_spawns
            spawn_counts_offset = 1 if self.initial_spawns == 1 else 0
            for i in range(self.current_step):
                remaining = actual - self.path[i]
                sc_index = i - spawn_counts_offset
                if sc_index < 0:
                    # First of the two 1→1 initial catches: field refills to 1 inherently
                    actual = max(1, remaining)
                elif sc_index < len(self.spawn_counts):
                    actual = max(self.spawn_counts[sc_index], remaining)
                else:
                    actual = remaining
            number_of_buttons = actual

        uncaught = []
        for row in range(self.path_table.rowCount()):
            if row not in self.stored_rows:
                uncaught.append(row)
            if len(uncaught) == number_of_buttons:
                break

        for counter, i in enumerate(uncaught, start=1):
            row_i = i
            if row_i >= self.path_table.rowCount():
                break
            if row_i not in self.stored_rows:
                btn = self.path_table.cellWidget(row_i, 3)
                if btn:
                    btn.setText(f"({counter})")
                    shortcut = getattr(Qt, f"Key_{counter}")
                    btn.setShortcut(shortcut)

    def _mo_field_at_step(self, step):
        """Return the visible catchable field size at the start of path step `step` for is_mo.

        For normal steps this equals min(4, first_wave_count - total_caught_before_step).
        For ghost/clear-wave steps, field refills normally up to the first ghost batch, then
        each subsequent ghost step depletes it by its actual catch count (ghosts respawn
        invisibly so the visible field shrinks with no new visible pokemon).
        """
        # Walk back to find the beginning of the current ghost/clear-wave chain.
        ghost_start = step
        while ghost_start > 0 and self.path[ghost_start - 1] > 10:
            ghost_start -= 1

        # Total actual catches before ghost_start (field refilled normally up to here).
        actual_kos_before = sum(self.pre_path) + sum(
            (v - 10 if (v > 10 and v != 255) else v)
            for v in self.path[:ghost_start]
        )
        field = min(4, self.first_wave_count - actual_kos_before)

        # Deduct each ghost step's catches (no visible respawns during the ghost chain).
        for g in range(ghost_start, step):
            ghost_val = self.path[g]
            ghost_kos = ghost_val - 10 if (ghost_val > 10 and ghost_val != 255) else ghost_val
            field = max(0, field - ghost_kos)

        return field

    def _step_ko_necessary(self, step):
        """Return the number of catches required to complete a given path step."""
        val = self.path[step]
        if val > 10 and val != 255:
            return val - 10   # ghost: actual catch count encoded as val-10
        elif val == 255:
            # Clear Wave: all remaining visible pokemon must be caught/KO'd.
            return self._mo_field_at_step(step)
        return val

    def on_caught_clicked(self, row_i):
        """Adds or remove the check mark to the caught button, and stores the row temporarily without lock"""
        if self.is_flashing:
            return
        if row_i in self.stored_rows:
            self.current_ko_count -= 1
            self.toggle_caught_button(True, row_i)
            del self.stored_rows[row_i]
            return
        max_ko_count = self._step_ko_necessary(self.current_step) if self.is_mo else self.path[self.current_step]
        if self.current_ko_count >= max_ko_count:
            self.flash_path_step()
            return

        if row_i not in self.stored_rows:
            self.stored_rows[row_i] = {
                'locked': False,
                'advance': self.current_step,
                'caught_number': self.caught_index,
                'button_text': "✓",
                'stored_time': self.current_time,
                'stored_weather': self.current_weather,
                'col_values': self.get_row_values(row_i),
            }
        self.current_ko_count += 1
        self.caught_index += 1
        self.toggle_caught_button(False, row_i)

    def toggle_caught_button(self, enabled, row):
        """Switches the buttons that can be pressed to reassign the shortcut and change the text on them from '[n] ✓' to '✓'"""
        btn = self.path_table.cellWidget(row, 3)
        time_item = self.path_table.item(row, 4)
        weather_item = self.path_table.item(row, 5)

        if enabled and btn is not None:
            # Remove the ✓ mark
            btn_text = btn.text()
            btn_number = btn_text[1]
            btn.setText("(" + btn_number + ")")
            shortcut = getattr(Qt, f"Key_{btn_number}")
            btn.setShortcut(shortcut)

            # Remove the time and weather icons
            if time_item:
                time_item.setData(Qt.DecorationRole, None)
            if weather_item:
                weather_item.setData(Qt.DecorationRole, None)

            # Remove any background
            for col in range(self.path_table.columnCount()):
                item = self.path_table.item(row, col)
                if item:
                    item.setBackground(Qt.NoBrush)
            
        else:
            # Add the ✓ mark
            btn_text = btn.text()
            btn_number = btn_text[1]
            btn.setText(btn_text + " ✓")

            # Add the time and weather icons
            if time_item:
                time_item.setData(Qt.DecorationRole, self.get_time_icon(self.current_time))
            if weather_item:
                weather_item.setData(Qt.DecorationRole, self.get_weather_icon(self.current_weather))

            # Add the background color
            for col in range(self.path_table.columnCount()):
                item = self.path_table.item(row, col)
                if item:
                    item.setBackground(QColor(35,55,75))

        # And reassign the shortcut in any case
        shortcut = getattr(Qt, f"Key_{btn_number}")
        btn.setShortcut(shortcut)

    def on_done(self):
        """Locks all the rows inside the stored rows (including those that were not locked) and proceeds to the next step"""
        if self.is_flashing:
            return
        current_ko_count_necessary = self._step_ko_necessary(self.current_step) if self.is_mo else self.path[self.current_step]
        if self.current_ko_count != current_ko_count_necessary:
            self.flash_path_step()
        else:
            # Change to ✓ and locked for all rows in the stored_rows
            for r in self.stored_rows:
                self.stored_rows[r]['locked'] = True
                self.stored_rows[r]['button_text'] = "✓"
            for row_i in self.stored_rows:
                btn = self.path_table.cellWidget(row_i, 3)
                if btn is not None:
                    btn.setText("✓")
                    btn.setShortcut(0)
                for col in range(self.path_table.columnCount()):
                    item = self.path_table.item(row_i, col)
                    if item:
                        item.setBackground(QColor(35,55,75))
            self.current_step += 1
            if self.current_step > len(self.path):
                self.current_step -= 1
            else:
                self.current_ko_count = 0
                self.renumber_step_buttons()
                self.update_path_display(self.path, self.current_step)

    def on_undo(self):
        """Removes all non-locked rows and moves to the previous step"""
        if self.is_flashing:
            return
        to_remove = [r for r, data in self.stored_rows.items() if data.get('caught_number') is not None and int(data['advance']) >= self.current_step - 1 and data['caught_number'] >= self.initial_number_to_store]
        for r in to_remove:
            del self.stored_rows[r]
        if self.current_step > self.step_correction:
            self.current_step -= 1
            self.current_ko_count = 0
            self.update_caught_index(type='undo')
            self.run_simulation(scroll_back_up=False)

    def on_reset(self):
        """Resets the path from the start"""
        if self.is_flashing:
            return
        to_remove = [r for r, data in self.stored_rows.items() if data.get('caught_number') is not None and data['caught_number'] >= self.initial_number_to_store]
        for r in to_remove:
            del self.stored_rows[r]
        self.current_step = self.step_correction
        self.current_ko_count = 0
        self.run_simulation(scroll_back_up=True)

    def change_time(self, time_val):
        """Handles the changing time interaction: removes all non-locked rows and changes the time to be stored when caught"""
        if self.is_flashing:
            return
        self.current_time = time_val
        if not self._initializing:
            to_remove = [r for r, data in self.stored_rows.items() if not data.get('locked', False)]
            for r in to_remove:
                del self.stored_rows[r]
            self.current_ko_count = 0
            self.run_simulation(scroll_back_up=False)

    def change_weather(self, weather_val):
        """Handles the changing time interaction: removes all non-locked rows and changes the weather to be stored when caught"""
        if self.is_flashing:
            return
        self.current_weather = weather_val
        if not self._initializing:
            to_remove = [r for r, data in self.stored_rows.items() if not data.get('locked', False)]
            for r in to_remove:
                del self.stored_rows[r]
            self.current_ko_count = 0
            self.run_simulation(scroll_back_up=False)

    def flash_path_step(self):
        """When the user wants to catch more pokemon than necessary at that step or not enough, the program will display a visual indication of the path to remind the user"""
        self.is_flashing = True
        original = self.path_display_label.text()
        parts = [str(s) for s in self.path] + ['Result']
        idx = self.current_step  # because step index = advance - first
        if 0 <= idx <= len(parts):
            parts[idx] = f'<big><b><span style="color:red;">{parts[idx]}</span></b></big>'
        blink = f'Path: {" → ".join(parts)}'
        def restore():
            self.path_display_label.setText(original)
            self.is_flashing = False
        QTimer.singleShot(250, lambda: self.path_display_label.setText(blink))
        QTimer.singleShot(500, lambda: self.path_display_label.setText(original))
        QTimer.singleShot(750, lambda: self.path_display_label.setText(blink))
        QTimer.singleShot(1000, lambda: restore())

    def update_last_advance_weather(self):
        """Store the time of day and weather for the last advance (the pokemon that the user chose to search for)"""
        for condition in self.last_advance_conditions.values():
            adv = condition['advance']
            time = condition['stored_time']
            weather = condition['stored_weather']
            for row in range(self.path_table.rowCount()):
                item = self.path_table.item(row, 0)
                if item is not None and item.text() == "Second wave spawns":
                    continue
                if item is not None and isinstance(adv, numbers.Number) and int(item.text()) == int(adv):
                    time_item = self.path_table.item(row, 4)
                    weather_item = self.path_table.item(row, 5)
                    if time_item:
                        time_item.setData(Qt.DecorationRole, self.get_time_icon(time))
                    if weather_item:
                        weather_item.setData(Qt.DecorationRole, self.get_weather_icon(weather))

    def update_path_display(self, path_tuple, idx):
        def label(n):
            if n < 10:
                return str(n)
            elif n == 255:
                return "Clear Wave"
            elif n < 20:
                return f"Ghost {n - 10}"
            return "Invalid"
        parts = [label(s) for s in path_tuple] + ["Result"]
        if 0 <= idx <= len(parts):
            parts[idx] = f'<big><b><span style="color:orange;">{parts[idx]}</span></b></big>'
        self.path_display_label.setText(f"Path: {' → '.join(parts)}")

    # ----------------------------------------------------------------------
    # Icon helpers
    # ----------------------------------------------------------------------
    def get_icon_path(self, name):
        return str(Path(__file__).parent.parent / "Resources" / "Icons" / name)

    def get_time_icon(self, val):
        if val == TIME_DAYTIME:
            return QIcon(self.get_icon_path("time_daytime.png"))
        if val == TIME_ANYTIME:
            return QIcon(self.get_icon_path("time_anytime.png"))
        try:
            name = LATime(val).name.lower()
        except:
            return None
        icons = {'dawn':'time_dawn.png','day':'time_day.png','dusk':'time_dusk.png','night':'time_night.png'}
        fname = icons.get(name)
        return QIcon(self.get_icon_path(fname)) if fname else None

    def get_weather_icon(self, val):
        try:
            name = LAWeather(val).name.lower()
        except:
            return None
        icons = {'sunny':'weather_sunny.png','cloudy':'weather_cloudy.png','rain':'weather_rain.png',
                 'snow':'weather_snow.png','drought':'weather_drought.png','fog':'weather_fog.png',
                 'rainstorm':'weather_rainstorm.png','snowstorm':'weather_snowstorm.png','none':'weather_any.png'}
        fname = icons.get(name)
        return QIcon(self.get_icon_path(fname)) if fname else None
    
class DoubleBorderDelegate(QStyledItemDelegate):
    """Delegate that draws a double line at the bottom of a specified row."""
    def __init__(self, parent=None, separator_row=-1):
        super().__init__(parent)
        self.separator_row = separator_row   # row index to highlight

    def paint(self, painter, option, index):
        # Draw the normal content first
        super().paint(painter, option, index)

        # If this is the separator row, paint a double line at the bottom
        if index.row() == self.separator_row and self.separator_row != -1:
            rect = option.rect
            # Draw two parallel lines (adjust Y positions and thickness as desired)
            pen = QPen(painter.pen())
            pen.setWidth(1)
            pen.setColor(QColor(120, 120, 120))   # grey
            painter.setPen(pen)

            # First line: at the very bottom of the cell
            y1 = rect.top() - 1
            painter.drawLine(rect.left(), y1, rect.right(), y1)

            # Second line: 2 pixels above the first (so they don't merge)
            y2 = rect.top() - 3
            painter.drawLine(rect.left(), y2, rect.right(), y2)