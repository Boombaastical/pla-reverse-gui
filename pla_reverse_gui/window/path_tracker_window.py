import numpy as np
from numba_pokemon_prngs.data.encounter import EncounterAreaLA
from numba_pokemon_prngs.data import NATURES_EN
from numba_pokemon_prngs.enums import LATime, LAWeather, LAArea
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection

from qtpy.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QButtonGroup, QDialog, QLabel, QPushButton,
    QSizePolicy, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QWidget,
)
from qtpy.QtGui import QIcon, QColor
from qtpy.QtCore import Qt, QSize, QTimer
from pathlib import Path
from functools import partial

from ..util import calc_effort_level, get_personal_index, get_name_en, path_to_string, AREA_WEATHERS
from ..pla_reverse_main.pla_reverse.size import calc_display_size

TIME_DAYTIME = 100
TIME_ANYTIME = 101

class PathTableWidget(QTableWidget):
    COLUMNS = (
        ("0xAdvance", 0),   # hidden, not used
        ("StepInAdv", 0),   # hidden, not used
        ("Locked", 0),      # hidden, not used
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
        self.setColumnCount(16)
        self.setHorizontalHeaderLabels([c[0] for c in self.COLUMNS])
        for i, (_, w) in enumerate(self.COLUMNS):
            self.setColumnWidth(i, w)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.verticalHeader().setVisible(False)
        # Optional: hide columns 0-2
        for i in range(3):
            self.setColumnHidden(i, True)
        delegate = IconDelegate(self, icon_size=32)
        self.setItemDelegateForColumn(4, delegate)  # Time column (index 4)
        self.setItemDelegateForColumn(5, delegate)  # Weather column (index 5)
        self.setMinimumHeight(500)
        self.setMinimumWidth(900)

class IconDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, icon_size=28):
        super().__init__(parent)
        self.icon_size = icon_size
    def paint(self, painter, option, index):
        icon = index.data(Qt.DecorationRole)
        if icon and isinstance(icon, QIcon):
            rect = option.rect
            pixmap = icon.pixmap(self.icon_size, self.icon_size)
            x = rect.x() + (rect.width() - pixmap.width()) // 2
            y = rect.y() + (rect.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)
        else:
            super().paint(painter, option, index)

class PathTrackerWindow(QDialog):
    def __init__(self, parent, encounter_table, second_wave_encounter_table, seed,
                 pre_path, path, count_values, max_spawn_count, weather, time,
                 species_info, spawn_counts, initial_spawns=0, area=None, allow_other_starts=False):
        super().__init__(parent)
        # store parameters
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

        # persistent storage: row_index -> dict
        self.stored_rows = {}

        # per‑simulation metadata
        self.row_advance = []      # index -> advance
        self.row_idx_in_step = []  # index -> idx_in_step
        self.step_ko_counts = {}   # advance -> total rows
        self.step_rows = {}           # advance -> list of row indices

        self.setWindowTitle("Path Tracker " + path_to_string(path))
        self.main_layout = QVBoxLayout(self)
        self.path_table = PathTableWidget()
        self._initializing = True

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
        self.btn_done.setEnabled(False)

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
        self.main_layout.addWidget(control_row)
        self.main_layout.addWidget(self.path_table)
        self.resize(sum(c[1] for c in self.path_table.COLUMNS), self.height())

        self._initializing = False
        self.run_simulation()

    # ----------------------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------------------
    def run_simulation(self):
        self.path_table.setRowCount(0)
        self.row_advance.clear()
        self.row_idx_in_step.clear()
        self.step_ko_counts.clear()
        self.step_rows.clear()

        current_encounter_table = self.encounter_table
        group_rng = Xoroshiro128PlusRejection(self.seed)
        ghost_count = 3

        count_vals = self.count_values
        if count_vals[0] != -1:
            full_sequence = (self.initial_spawns,) + self.path
            count_vals = (self.initial_spawns,) + self.pre_path + self.count_values
        else:
            full_sequence = self.pre_path + self.path

        pre_len = len(self.pre_path)
        index_start_modifier = pre_len if not self.allow_other_starts else 0

        for advance, spawn_count in enumerate(full_sequence):
            is_ghost = False
            # special values
            if spawn_count == 255:
                spawn_count = 4
                current_encounter_table = self.second_wave_encounter_table
            elif spawn_count > 20:
                continue
            elif spawn_count > 10:
                ghost_count -= spawn_count - 10
                spawn_count = 3 - ghost_count
                is_ghost = True

            # build displayed path
            if advance < pre_len:
                current_path = ()
            else:
                conditional = len(self.pre_path) - 1 if len(self.pre_path) > 1 else 1
                current_path = full_sequence[1 : advance + conditional]

            # variable multi logic
            if count_vals[0] != -1:
                cur_spawn = count_vals[advance]
                before = cur_spawn - spawn_count
                next_spawn = count_vals[advance + 1]
                generated = max(0, next_spawn - before)
                spawn_count = generated

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
                caught_btn.setFixedSize(40,25)
                self.path_table.setCellWidget(row_i, 3, caught_btn)

                # Time (col4) and Weather (col5) items
                time_item = QTableWidgetItem()
                weather_item = QTableWidgetItem()
                time_item.setText("")
                weather_item.setText("")
                self.path_table.setItem(row_i, 4, time_item)
                self.path_table.setItem(row_i, 5, weather_item)

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
                for j, val in enumerate(row_data, start=6):
                    self.path_table.setItem(row_i, j, QTableWidgetItem(val))

                # store metadata
                self.row_advance.append(display_adv)
                self.row_idx_in_step.append(idx_in_step)
                self.step_ko_counts[display_adv] = self.step_ko_counts.get(display_adv, 0) + 1
                self.step_rows.setdefault(display_adv, []).append(row_i)

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
            print(f"Row: {row_i}, Stored values: {stored}")

        # --- lock negative advances and last advance ---
        if self.row_advance:
            max_adv = max(self.row_advance)
            for row_i, adv in enumerate(self.row_advance):
                if adv < 0 and row_i not in self.stored_rows:
                    self.stored_rows[row_i] = {
                        'locked': True,
                        'caught_number': -1,
                        'button_text': "✓",
                        'stored_time': self.current_time,
                        'stored_weather': self.current_weather,
                        'col_values': self.get_row_values(row_i),
                    }
                    btn = self.path_table.cellWidget(row_i, 3)
                    if btn:
                        btn.setText("✓")
                    for col in range(self.path_table.columnCount()):
                        item = self.path_table.item(row_i, col)
                        if item:
                            item.setBackground(QColor(35,55,75))
                elif adv == max_adv and row_i not in self.stored_rows:
                    self.stored_rows[row_i] = {
                        'locked': True,
                        'caught_number': None,
                        'button_text': "",
                        'stored_time': self.current_time,
                        'stored_weather': self.current_weather,
                        'col_values': self.get_row_values(row_i),
                    }
                    for col in range(self.path_table.columnCount()):
                        item = self.path_table.item(row_i, col)
                        if item:
                            item.setBackground(QColor(35,55,75))

        # --- determine current step ---
        first_non_neg = min((a for a in self.step_ko_counts if a >= 0), default=None)
        self.current_step = 0
        if first_non_neg is not None:
            for adv in sorted(self.step_ko_counts.keys()):
                if adv < 0:
                    continue
                step_idx = adv - first_non_neg
                total = self.step_ko_counts[adv]
                caught = sum(1 for r in self.step_rows[adv] 
                             if r in self.stored_rows and self.stored_rows[r].get('caught_number') == step_idx)
                if caught < total:
                    self.current_step = step_idx
                    break
            else:
                self.current_step = max(adv - first_non_neg for adv in self.step_ko_counts if adv >= 0)

        self.renumber_step_buttons()
        self.update_done_button_state()
        if first_non_neg is not None:
            self.update_path_display(self.path, self.current_step)

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def get_row_values(self, row_i):
        """Return list of text values for columns 6..15 (visible data)."""
        values = []
        for col in range(6, self.path_table.columnCount()):
            item = self.path_table.item(row_i, col)
            values.append(item.text() if item else "")
        return values

    def renumber_step_buttons(self):
        # Clear all button texts on rows that are not caught
        for row_i in range(self.path_table.rowCount()):
            if row_i not in self.stored_rows:
                btn = self.path_table.cellWidget(row_i, 3)
                if btn:
                    btn.setText("")
                    btn.setShortcut(0)  # remove shortcut

        if self.spawn_counts[0] == -1:
            # Non‑variable spawner (regular multispawner, MO, MMO)
            # Numbers 1..max_spawn_count are fixed to the first max_spawn_count rows.
            for i in range(1, self.max_spawn_count + 1):
                row_i = i - 1
                if row_i >= self.path_table.rowCount():
                    break
                if row_i not in self.stored_rows:
                    btn = self.path_table.cellWidget(row_i, 3)
                    if btn:
                        btn.setText(f"({i})")
                        if i <= 9:
                            shortcut = getattr(Qt, f"Key_{i}")
                            btn.setShortcut(shortcut)
        else:
            # Variable spawner – step‑based numbering
            first_non_neg = min((a for a in self.step_ko_counts if a >= 0), default=None)
            if first_non_neg is None:
                return
            cur_adv = first_non_neg + self.current_step
            if cur_adv not in self.step_rows:
                return
            if self.current_step >= len(self.spawn_counts):
                return
            target = self.spawn_counts[self.current_step]
            # Rows in this step, ordered by their natural order (idx_in_step)
            step_rows = sorted(self.step_rows[cur_adv], key=lambda r: self.row_idx_in_step[r])
            # Assign numbers to the first 'target' uncaught rows in this step
            assigned = 0
            for r in step_rows:
                if r not in self.stored_rows and assigned < target:
                    assigned += 1
                    btn = self.path_table.cellWidget(r, 3)
                    if btn:
                        btn.setText(f"({assigned})")
                        if assigned <= 9:
                            shortcut = getattr(Qt, f"Key_{assigned}")
                            btn.setShortcut(shortcut)
                elif r not in self.stored_rows:
                    # Extra rows in this step (beyond target) remain unnumbered
                    btn = self.path_table.cellWidget(r, 3)
                    if btn:
                        btn.setText("")

    def update_done_button_state(self):
        first = min((a for a in self.step_ko_counts if a >= 0), default=None)
        if first is None:
            self.btn_done.setEnabled(False)
            return
        cur_adv = first + self.current_step
        total = self.step_ko_counts.get(cur_adv, 0)
        caught = sum(1 for r in self.step_rows.get(cur_adv, []) if r in self.stored_rows)
        self.btn_done.setEnabled(caught == total)

    def on_caught_clicked(self, row_i):
        if row_i in self.stored_rows:
            return
        adv = self.row_advance[row_i]
        first = min((a for a in self.step_ko_counts if a >= 0), default=None)
        if adv < 0:
            return
        step_idx = adv - first
        if step_idx != self.current_step:
            self.flash_path_step()
            return
        btn = self.path_table.cellWidget(row_i, 3)
        current_text = btn.text() if btn else ""
        self.stored_rows[row_i] = {
            'locked': False,
            'caught_number': step_idx,
            'button_text': current_text,
            'stored_time': self.current_time,
            'stored_weather': self.current_weather,
            'col_values': self.get_row_values(row_i),
        }
        if btn:
            btn.setText("✓")
        time_item = self.path_table.item(row_i, 4)
        weather_item = self.path_table.item(row_i, 5)
        if time_item:
            time_item.setData(Qt.DecorationRole, self.get_time_icon(self.current_time))
        if weather_item:
            weather_item.setData(Qt.DecorationRole, self.get_weather_icon(self.current_weather))
        for col in range(self.path_table.columnCount()):
            item = self.path_table.item(row_i, col)
            if item:
                item.setBackground(QColor(35,55,75))
        self.update_done_button_state()

    def on_done(self):
        if not self.btn_done.isEnabled():
            self.flash_path_step()
            return
        first = min((a for a in self.step_ko_counts if a >= 0), default=None)
        if first is None:
            return
        cur_adv = first + self.current_step
        for r in self.step_rows.get(cur_adv, []):
            if r in self.stored_rows and not self.stored_rows[r].get('locked', False):
                self.stored_rows[r]['locked'] = True
                for col in range(self.path_table.columnCount()):
                    item = self.path_table.item(r, col)
                    if item:
                        item.setBackground(QColor(35,55,75))
        self.current_step += 1
        next_adv = first + self.current_step
        if next_adv not in self.step_ko_counts:
            self.current_step -= 1
            self.btn_done.setEnabled(False)
        else:
            self.renumber_step_buttons()
            self.update_done_button_state()
            self.update_path_display(self.path, self.current_step)

    def on_undo(self):
        first = min((a for a in self.step_ko_counts if a >= 0), default=None)
        if first is None:
            return
        to_remove = [r for r, data in self.stored_rows.items() if data.get('caught_number') is not None and data['caught_number'] >= self.current_step]
        for r in to_remove:
            del self.stored_rows[r]
        if self.current_step > 0:
            self.current_step -= 1
        self.run_simulation()

    def on_reset(self):
        to_remove = [r for r, data in self.stored_rows.items() if data.get('caught_number') is not None and data['caught_number'] >= 0]
        for r in to_remove:
            del self.stored_rows[r]
        self.current_step = 0
        self.run_simulation()

    def change_time(self, time_val):
        self.current_time = time_val
        if not self._initializing:
            to_remove = [r for r, data in self.stored_rows.items() if not data.get('locked', False)]
            for r in to_remove:
                del self.stored_rows[r]
            self.run_simulation()

    def change_weather(self, weather_val):
        self.current_weather = weather_val
        if not self._initializing:
            to_remove = [r for r, data in self.stored_rows.items() if not data.get('locked', False)]
            for r in to_remove:
                del self.stored_rows[r]
            self.run_simulation()

    def flash_path_step(self):
        original = self.path_display_label.text()
        parts = [str(s) for s in self.path]
        first = min((a for a in self.step_ko_counts if a >= 0), default=0)
        idx = self.current_step  # because step index = advance - first
        if 0 <= idx < len(parts):
            parts[idx] = f'<span style="background-color:red;color:white;">{parts[idx]}</span>'
        blink = " → ".join(parts)
        self.path_display_label.setText(blink)
        QTimer.singleShot(500, lambda: self.path_display_label.setText(original))

    def flash_button(self, button):
        orig = button.styleSheet()
        button.setStyleSheet("background-color: lightblue;")
        QTimer.singleShot(200, lambda: button.setStyleSheet(orig))

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

    def update_path_display(self, path_tuple, idx):
        parts = [str(s) for s in path_tuple]
        if 0 <= idx < len(parts):
            parts[idx] = f"<big><b>{parts[idx]}</b></big>"
        self.path_display_label.setText(f"Path: {' → '.join(parts)}")