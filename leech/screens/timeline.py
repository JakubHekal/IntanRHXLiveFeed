import time

from PyQt5.QtCore import pyqtSignal, Qt, QRect, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics
from PyQt5.QtWidgets import QWidget, QMenu, QInputDialog, QMessageBox, QPushButton

from leech.experiment.migrations import SYSTEM_DEVICE_ID, SYSTEM_DEVICE_TYPE, new_device_id

from ._registry import _DEVICE_CLASSES, _SYSTEM_OPERATIONS


class DeviceRow(list):
    def __init__(self, name, blocks, device_type, config=None, device_id=None, config_version=1):
        super().__init__([name, blocks, device_type, dict(config or {})])
        self.device_id = device_id or (
            SYSTEM_DEVICE_ID if device_type == SYSTEM_DEVICE_TYPE else new_device_id()
        )
        self.config_version = config_version
        self.instance = None


class ExperimentTimeline(QWidget):
    ROW_HEIGHT = 36
    BLOCK_H = ROW_HEIGHT - 6
    LANE_OFF = 10
    FLAG_HEIGHT = 18
    LABEL_WIDTH = 130
    HEADER_HEIGHT = 28
    RESIZE_THRESHOLD = 6
    SNAP = 30.0
    _BLOCK_COLORS = ["#0078D4", "#2B88D8", "#4BA3E3", "#107C10", "#498205",
                     "#D13438", "#E74856", "#F1707A", "#8764B8", "#B146C2", "#C239B3"]

    block_selected = pyqtSignal(int, int, str, str, float, float, dict, str)
    data_changed = pyqtSignal()
    device_selected = pyqtSignal(int, str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._device_counter = 0
        self._devices = []
        self._sel_dev = None
        self._sel_block = None
        self._active_dev = None
        self._active_block = None
        self._drag_state = None
        self._drag_dev = None
        self._drag_block = None
        self._drag_press_x = 0.0
        self._drag_orig_start = 0.0
        self._drag_orig_dur = 0.0
        self._running = False
        self._edit_mode = True
        self._cursor_time = None
        self._cursor_step_start = 0.0
        self._cursor_step_end = 0.0
        self._cursor_wall_start = 0.0
        self._cursor_wall_dur = 1.0
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setInterval(30)
        self._cursor_timer.timeout.connect(self._on_cursor_tick)
        self.add_device_button = QPushButton("+ Add Device", self)
        self.add_device_button.setFocusPolicy(Qt.NoFocus)
        self.add_device_button.clicked.connect(self.add_device_dialog)
        self._position_header_button()
        self._update_total_time()
        self._update_height()

    def _update_height(self):
        h = self.HEADER_HEIGHT + 6 + sum(self._row_height(i) for i in range(len(self._devices))) + 16
        self.setMinimumHeight(max(h, 200))

    def _position_header_button(self):
        if not hasattr(self, "add_device_button"):
            return
        self.add_device_button.setGeometry(
            4,
            3,
            max(1, self.LABEL_WIDTH - 8),
            max(1, self.HEADER_HEIGHT - 6),
        )
        self.add_device_button.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_header_button()

    def _row_lanes(self, blocks):
        events = []
        for i, block in enumerate(blocks):
            start, dur = block[1], block[2]
            if dur == 0:
                continue
            events.append((start, start + dur, i))
        events.sort()
        lanes = {}
        lane_ends = []
        for start, end, i in events:
            for li, le in enumerate(lane_ends):
                if le <= start:
                    lane_ends[li] = end
                    lanes[i] = li
                    break
            else:
                lane_ends.append(end)
                lanes[i] = len(lane_ends) - 1
        return lanes, len(lane_ends)

    def _row_height(self, dev_idx):
        if self._devices[dev_idx][2] == "__system__":
            return self.FLAG_HEIGHT + self.ROW_HEIGHT
        _, lanes = self._row_lanes(self._devices[dev_idx][1])
        return self.FLAG_HEIGHT + max(self.ROW_HEIGHT, 33 + self.LANE_OFF * (lanes - 1))

    def _row_origins(self):
        origins = []
        y = self.HEADER_HEIGHT + 6
        for i in range(len(self._devices)):
            origins.append(y)
            y += self._row_height(i)
        return origins

    def _row_at(self, my):
        if my < self.HEADER_HEIGHT + 6:
            return None
        for dev_idx, row_origin in enumerate(self._row_origins()):
            if row_origin <= my < row_origin + self._row_height(dev_idx):
                return dev_idx
        return None

    def add_device(self, name=None, device_type=None, config=None, device_id=None, config_version=1):
        if not name:
            self._device_counter += 1
            name = f"Device {self._device_counter}"
        if device_type is None:
            device_type = "rhx"
        cls = _DEVICE_CLASSES.get(device_type)
        if config is None:
            config = {}
            if cls:
                config = {p.name: p.default for p in cls.get_config_params()}
        row = DeviceRow(name, [], device_type, config, device_id, config_version)
        self._devices.append(row)
        self._update_total_time()
        self._update_height()
        self.data_changed.emit()
        self.update()
        return row

    def add_system_device(self):
        if any(d[2] == SYSTEM_DEVICE_TYPE for d in self._devices):
            return next(d for d in self._devices if d[2] == SYSTEM_DEVICE_TYPE)
        row = DeviceRow("System actions", [], SYSTEM_DEVICE_TYPE, {}, SYSTEM_DEVICE_ID)
        self._devices.append(row)
        self._update_total_time()
        self._update_height()
        self.update()
        return row

    def add_device_dialog(self):
        if not self._edit_mode:
            return
        choices = sorted(_DEVICE_CLASSES.items(), key=lambda item: item[1].name)
        labels = [f"{cls.name} ({device_type})" for device_type, cls in choices]
        label, ok = QInputDialog.getItem(self, "Add Device", "Device type:", labels, 0, False)
        if not ok or not label:
            return
        device_type = next(
            device_type for device_type, cls in choices
            if f"{cls.name} ({device_type})" == label
        )
        cls = _DEVICE_CLASSES[device_type]
        name, ok = QInputDialog.getText(self, "Add Device", "Device name:", text=cls.name)
        if not ok:
            return
        name = name.strip() or cls.name
        if any(d[0] == name and d[2] != "__system__" for d in self._devices):
            QMessageBox.warning(self, "Device Exists", f"Device '{name}' is already in this plan.")
            return
        self.add_device(name, device_type)

    def _operations_for_device(self, device_type):
        if device_type == "__system__":
            return _SYSTEM_OPERATIONS
        cls = _DEVICE_CLASSES.get(device_type)
        return cls.get_operations() if cls else []

    def _device_name_at(self, mx, my):
        row_idx = self._row_at(my)
        if row_idx is None:
            return None
        row_y = self._row_origins()[row_idx] + self.FLAG_HEIGHT
        name_width = QFontMetrics(QFont("Segoe UI", 9)).horizontalAdvance(self._devices[row_idx][0])
        name_right = min(self.LABEL_WIDTH - 4, 8 + name_width + 4)
        if 8 <= mx <= name_right and row_y <= my < row_y + self.ROW_HEIGHT:
            return row_idx
        return None

    def set_edit_mode(self, enabled):
        self._edit_mode = bool(enabled)
        self.add_device_button.setVisible(self._edit_mode)
        self.update()

    def remove_device(self, dev_idx):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        if self._devices[dev_idx][2] == "__system__":
            return
        del self._devices[dev_idx]
        if self._sel_dev == dev_idx:
            self._sel_dev = None
            self._sel_block = None
        elif self._sel_dev is not None and self._sel_dev > dev_idx:
            self._sel_dev -= 1
        self._update_total_time()
        self._update_height()
        self.data_changed.emit()
        self.update()

    def clear_all(self):
        system_entry = None
        for d in self._devices:
            if d[2] == SYSTEM_DEVICE_TYPE:
                system_entry = DeviceRow(
                    d[0], [], d[2], d[3] if len(d) >= 4 else {},
                    getattr(d, "device_id", SYSTEM_DEVICE_ID),
                    getattr(d, "config_version", 1),
                )
                break
        self._devices.clear()
        if system_entry:
            self._devices.append(system_entry)
        self._sel_dev = None
        self._sel_block = None
        self._update_total_time()
        self._update_height()
        self.data_changed.emit()
        self.update()

    def set_active_block(self, device_name, block_label):
        for i, row in enumerate(self._devices):
            if row[2] == "__system__":
                continue
            if row[0] != device_name:
                continue
            for bi, block in enumerate(row[1]):
                if block[0] == block_label:
                    self._active_dev = i
                    self._active_block = bi
                    start, dur = block[1], block[2]
                    if start is not None and dur is not None:
                        self._cursor_step_start = start
                        self._cursor_step_end = start + dur
                        self._cursor_wall_start = time.perf_counter()
                        self._cursor_wall_dur = dur if dur > 0 else 1.0
                        self._cursor_time = start
                        if not self._cursor_timer.isActive():
                            self._cursor_timer.start()
                    self.update()
                    return
        self._active_dev = None
        self._active_block = None
        self.update()

    def set_active_step(self, step_index):
        count = 0
        for i, row in enumerate(self._devices):
            if row[2] == "__system__":
                continue
            blocks = row[1]
            if step_index < count + len(blocks):
                self._active_dev = i
                self._active_block = step_index - count
                break
            count += len(blocks)
        else:
            self._active_dev = None
            self._active_block = None

        start, dur = self._step_timeline_range(step_index)
        if start is not None and dur is not None:
            self._cursor_step_start = start
            self._cursor_step_end = start + dur
            self._cursor_wall_start = time.perf_counter()
            self._cursor_wall_dur = dur if dur > 0 else 1.0
            self._cursor_time = start
            if not self._cursor_timer.isActive():
                self._cursor_timer.start()
        self.update()

    def clear_active_step(self):
        self._active_dev = None
        self._active_block = None
        self._cursor_timer.stop()
        self._cursor_time = None
        self.update()

    def add_block(self, dev_idx, op_name="New Block", start=None, duration=None, params=None):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        blocks = self._devices[dev_idx][1]
        if start is None:
            start = max(0.0, self._total_time - 2.0)
        label = op_name
        color = self._BLOCK_COLORS[len(blocks) % len(self._BLOCK_COLORS)]
        device_type = self._devices[dev_idx][2]
        device_class = _DEVICE_CLASSES.get(device_type)
        canonical_op_name = (
            device_class.canonical_operation_id(op_name)
            if device_class and hasattr(device_class, "canonical_operation_id")
            else op_name
        )
        ops = _SYSTEM_OPERATIONS if device_type == SYSTEM_DEVICE_TYPE else (getattr(device_class, 'get_operations', lambda: [])())
        op_duration = duration
        for op in ops:
            if op.operation_id == canonical_op_name:
                op_name = canonical_op_name
                label = params.pop("block_label", op.label) if params else op.label
                color = op.color
                if op.instantaneous:
                    op_duration = 0
                elif op_duration is None:
                    op_duration = op.default_duration
                if params is None:
                    params = {p.name: p.default for p in op.params}
                break
        if op_duration is None:
            op_duration = 2.0
        if params is None:
            params = {}
        blocks.append([label, start, op_duration, color, op_name, params])
        self._update_total_time()
        self.data_changed.emit()
        self.update()

    def remove_block(self, dev_idx, block_idx):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        blocks = self._devices[dev_idx][1]
        if block_idx < 0 or block_idx >= len(blocks):
            return
        del blocks[block_idx]
        if self._sel_dev == dev_idx and self._sel_block == block_idx:
            self._sel_block = None
        elif self._sel_dev == dev_idx and self._sel_block is not None and self._sel_block > block_idx:
            self._sel_block -= 1
        self._update_total_time()
        self.data_changed.emit()
        self.update()

    def duplicate_block(self, dev_idx, block_idx):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        blocks = self._devices[dev_idx][1]
        if block_idx < 0 or block_idx >= len(blocks):
            return
        orig = blocks[block_idx]
        new_start = orig[1] + (orig[2] if orig[2] > 0 else self.SNAP)
        dup = [orig[0], new_start, orig[2], orig[3], orig[4],
               dict(orig[5]) if len(orig) >= 6 and isinstance(orig[5], dict) else {}]
        blocks.insert(block_idx + 1, dup)
        self._update_total_time()
        self.data_changed.emit()
        self.update()

    def _build_add_block_menu(self, parent_menu, target_dev):
        sub = QMenu("Add Block", parent_menu)
        actions = {}
        for op in self._operations_for_device(self._devices[target_dev][2]):
            action = sub.addAction(op.label)
            actions[action] = op.operation_id
        if not actions:
            action = sub.addAction("Generic Block")
            actions[action] = "New Block"
        return sub, actions

    def contextMenuEvent(self, event):
        if not self._edit_mode:
            return
        mx, my = event.x(), event.y()
        menu = QMenu(self)
        dev_idx, block_idx, _ = self._block_at(mx, my)
        row_idx = self._row_at(my)
        name_idx = self._device_name_at(mx, my)
        is_system_row = row_idx is not None and self._devices[row_idx][2] == "__system__"

        if block_idx is not None:
            a_dup = menu.addAction(f"Duplicate  «{self._devices[dev_idx][1][block_idx][0]}»")
            a_del = menu.addAction(f"Remove  «{self._devices[dev_idx][1][block_idx][0]}»")
            menu.addSeparator()
            add_menu, add_actions = self._build_add_block_menu(menu, dev_idx)
            menu.addMenu(add_menu)
            action = menu.exec_(event.globalPos())
            if action == a_dup:
                self.duplicate_block(dev_idx, block_idx)
            elif action == a_del:
                self.remove_block(dev_idx, block_idx)
            elif action in add_actions:
                self.add_block(dev_idx, add_actions[action])
        elif row_idx is not None:
            add_menu, add_actions = self._build_add_block_menu(menu, row_idx)
            menu.addMenu(add_menu)
            if name_idx == row_idx and not is_system_row:
                a_del_d = menu.addAction(f"Remove Device  «{self._devices[row_idx][0]}»")
            action = menu.exec_(event.globalPos())
            if action in add_actions:
                self.add_block(row_idx, add_actions[action])
            elif name_idx == row_idx and not is_system_row and action == a_del_d:
                reply = QMessageBox.question(
                    self,
                    "Remove Device",
                    f"Remove device '{self._devices[row_idx][0]}' from this experiment?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self.remove_device(row_idx)
        else:
            add_dev = menu.addAction("Add Device")
            sys_menu = QMenu("Add System Block", menu)
            sys_actions = {}
            for op in _SYSTEM_OPERATIONS:
                action = sys_menu.addAction(op.label)
                sys_actions[action] = op.operation_id
            menu.addMenu(sys_menu)
            action = menu.exec_(event.globalPos())
            if action == add_dev:
                self.add_device_dialog()
            elif action in sys_actions:
                system_row = self.add_system_device()
                self.add_block(self._devices.index(system_row), sys_actions[action])


    def _update_total_time(self):
        self._total_time = max(
            (s + d for row in self._devices for _, s, d, *_ in row[1]),
            default=1200,
        )
        if self._total_time <= 0:
            self._total_time = 1200

    def set_running(self, running):
        self._running = running
        self.set_edit_mode(not running)
        if not running:
            self._cursor_timer.stop()
            self._cursor_time = None
            self.update()

    def _step_timeline_range(self, step_index):
        # ponytail: iterate in device order matching set_active_step counting,
        # not sorted by start time — interleaved blocks from different devices
        # would map to wrong positions when sorted.
        count = 0
        for row in self._devices:
            if row[2] == "__system__":
                continue
            for block in row[1]:
                if count == step_index:
                    return block[1], block[2]  # start, dur
                count += 1
        return None, None

    def _on_cursor_tick(self):
        elapsed = time.perf_counter() - self._cursor_wall_start
        frac = min(1.0, elapsed / self._cursor_wall_dur) if self._cursor_wall_dur > 0 else 1.0
        self._cursor_time = self._cursor_step_start + frac * (self._cursor_step_end - self._cursor_step_start)
        self.update()

    def _plot_left(self):
        return self.LABEL_WIDTH

    def _plot_w(self):
        return max(1, self.width() - self._plot_left() - 12)

    def _x_from_time(self, t):
        return self._plot_left() + int((t / self._total_time) * self._plot_w())

    def _snap(self, t):
        return round(t / self.SNAP) * self.SNAP

    def _block_at(self, mx, my):
        if my < self.HEADER_HEIGHT + 6:
            return None, None, None
        origins = self._row_origins()
        dev_idx = None
        for i, row_origin in enumerate(origins):
            if row_origin <= my < row_origin + self._row_height(i):
                dev_idx = i
                break
        if dev_idx is None:
            return None, None, None
        blocks = self._devices[dev_idx][1]
        row_origin = origins[dev_idx]
        in_flag = my < row_origin + self.FLAG_HEIGHT
        lanes, _ = self._row_lanes(blocks) if not in_flag else ({}, 0)
        for bi in sorted(range(len(blocks)), key=lambda i: (lanes.get(i, 0), i), reverse=True):
            func, start, dur, *_ = blocks[bi]
            is_instant = dur == 0
            if in_flag != is_instant:
                continue
            bx = self._x_from_time(start)
            if is_instant:
                fm = QFontMetrics(QFont("Segoe UI", 8))
                pill_w = max(24, fm.horizontalAdvance(func) + 12)
                pill_x = max(self._plot_left(), bx - pill_w // 2)
                if pill_x <= mx <= pill_x + pill_w and row_origin <= my < row_origin + self.FLAG_HEIGHT:
                    return dev_idx, bi, "body"
            else:
                by = row_origin + self.FLAG_HEIGHT + 3 + lanes.get(bi, 0) * self.LANE_OFF
                bh = self.BLOCK_H
                bw = max(4, int((dur / self._total_time) * self._plot_w()))
                if (bx - self.RESIZE_THRESHOLD <= mx <= bx + bw + self.RESIZE_THRESHOLD
                        and by <= my < by + bh):
                    if dur > 0 and abs(mx - bx) <= self.RESIZE_THRESHOLD:
                        return dev_idx, bi, "left"
                    if dur > 0 and abs(mx - (bx + bw)) <= self.RESIZE_THRESHOLD:
                        return dev_idx, bi, "right"
                    return dev_idx, bi, "body"
        return None, None, None

    def _select(self, dev_idx, block_idx):
        self._sel_dev = dev_idx
        self._sel_block = block_idx
        if dev_idx is not None and block_idx is not None and block_idx < len(self._devices[dev_idx][1]):
            blocks = self._devices[dev_idx][1]
            b = blocks[block_idx]
            func = b[0]
            start = b[1]
            dur = b[2]
            op_name = b[4] if len(b) >= 5 else ""
            params = b[5] if len(b) >= 6 else {}
            device_type = self._devices[dev_idx][2]
            self.block_selected.emit(dev_idx, block_idx, func, op_name, start, dur, params, device_type)
        elif dev_idx is not None:
            device_type = self._devices[dev_idx][2]
            config = self._devices[dev_idx][3] if len(self._devices[dev_idx]) >= 4 else {}
            self.device_selected.emit(dev_idx, device_type, config)
        else:
            self.block_selected.emit(-1, -1, "", "", 0.0, 0.0, {}, "")
            self.device_selected.emit(-1, "", {})
        self.update()

    def update_block(self, dev_idx, block_idx, func_name, start, duration, params=None):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        blocks = self._devices[dev_idx][1]
        if block_idx < 0 or block_idx >= len(blocks):
            return
        is_instant = (duration == 0)
        if not is_instant:
            duration = max(0.5, duration)
        start = max(0.0, start)
        if is_instant:
            duration = 0
        blocks[block_idx][0] = func_name
        blocks[block_idx][1] = start
        blocks[block_idx][2] = duration
        if params is not None:
            if len(blocks[block_idx]) < 6:
                blocks[block_idx].append("")
                blocks[block_idx].append({})
            blocks[block_idx][5] = params
        self._update_total_time()
        self.data_changed.emit()
        self.update()

    def update_device_config(self, dev_idx, config):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        while len(self._devices[dev_idx]) < 4:
            self._devices[dev_idx].append({})
        self._devices[dev_idx][3] = config
        self.data_changed.emit()

    def mousePressEvent(self, event):
        if not self._edit_mode:
            return
        if event.button() != Qt.LeftButton:
            return
        mx, my = event.x(), event.y()
        dev_idx, block_idx, edge = self._block_at(mx, my)
        if dev_idx is not None and block_idx is not None:
            self._select(dev_idx, block_idx)
            blocks = self._devices[dev_idx][1]
            self._drag_state = f"resize_{edge}" if edge in ("left", "right") else "move"
            self._drag_dev = dev_idx
            self._drag_block = block_idx
            self._drag_press_x = mx
            self._drag_orig_start = blocks[block_idx][1]
            self._drag_orig_dur = blocks[block_idx][2]
        else:
            row_idx = self._row_at(my)
            if row_idx is not None:
                self._select(row_idx, None)
            else:
                self._select(None, None)

    def mouseMoveEvent(self, event):
        mx = event.x()
        my = event.y()
        if not self._edit_mode:
            self.setCursor(Qt.ArrowCursor)
            return
        if self._drag_state:
            if self._drag_state in ("resize_left", "resize_right"):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.SizeAllCursor)
            blocks = self._devices[self._drag_dev][1]
            block = blocks[self._drag_block]
            total = self._total_time
            pw = self._plot_w()
            dt = ((mx - self._drag_press_x) / pw) * total if pw > 0 else 0.0

            if self._drag_state == "move":
                new_start = self._snap(max(0.0, min(
                    self._drag_orig_start + dt,
                    total - block[2],
                )))
                block[1] = new_start
            elif self._drag_state == "resize_left":
                new_start = self._snap(max(0.0, min(
                    self._drag_orig_start + dt,
                    self._drag_orig_start + self._drag_orig_dur - self.SNAP,
                )))
                new_dur = self._snap(max(
                    self.SNAP,
                    self._drag_orig_start + self._drag_orig_dur - new_start,
                ))
                block[1] = new_start
                block[2] = new_dur
            elif self._drag_state == "resize_right":
                new_dur = self._snap(max(
                    self.SNAP,
                    min(self._drag_orig_dur + dt, total - self._drag_orig_start),
                ))
                block[2] = new_dur

            self._update_total_time()
            self.data_changed.emit()
            block_op = block[4] if len(block) >= 5 else ""
            block_params = block[5] if len(block) >= 6 else {}
            device_type = self._devices[self._drag_dev][2] if self._drag_dev is not None and self._drag_dev < len(self._devices) else ""
            self.block_selected.emit(
                self._drag_dev, self._drag_block,
                block[0], block_op, block[1], block[2], block_params, device_type,
            )
            self.update()
        else:
            _, _, edge = self._block_at(mx, my)
            if edge in ("left", "right"):
                self.setCursor(Qt.SizeHorCursor)
            elif edge == "body":
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if not self._edit_mode:
            return
        self._drag_state = None
        self._drag_dev = None
        self._drag_block = None
        self._drag_press_x = 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w = rect.width()

        painter.fillRect(rect, QColor("#1E1E1E"))

        plot_left = self.LABEL_WIDTH
        plot_w = max(1, w - plot_left - 12)

        # Time header
        painter.setPen(QPen(QColor("#EDEBE9"), 1))
        painter.setFont(QFont("Segoe UI", 9))
        painter.fillRect(0, 0, w, self.HEADER_HEIGHT, QColor("#2D2D2D"))

        min_gap = 80
        step = max(2, int(self._total_time * min_gap / plot_w))
        for t in range(0, int(self._total_time) + 1, step):
            x = plot_left + int((t / self._total_time) * plot_w)
            painter.drawLine(x, self.HEADER_HEIGHT - 4, x, self.HEADER_HEIGHT)
            painter.drawText(x - 10, self.HEADER_HEIGHT - 8, str(t))

        rows_top = self.HEADER_HEIGHT + 6
        origins = self._row_origins()
        rows_height = sum(self._row_height(i) for i in range(len(self._devices)))
        empty_plan = not any(d[2] != "__system__" or d[1] for d in self._devices)
        if empty_plan:
            fnt = self.font()
            fnt.setPointSize(11)
            painter.setFont(fnt)
            painter.setPen(QColor("#6C6C6C"))
            data_rect = QRect(0, rows_top, w, max(40, self.height() - rows_top - 80))
            painter.drawText(
                data_rect,
                Qt.AlignCenter,
                "No devices yet.\nUse + Add Device in the timeline header.",
            )

        # Phase 2: Draw non-system device rows
        for i, row in enumerate(self._devices):
            name, blocks, device_type = row[0], row[1], row[2]
            if device_type == "__system__":
                continue
            flag_y = origins[i]
            row_y = flag_y + self.FLAG_HEIGHT

            bg = QColor("#252526") if i % 2 == 0 else QColor("#1E1E1E")
            painter.fillRect(0, flag_y, w, self.FLAG_HEIGHT, bg)
            painter.fillRect(0, row_y, w, self._row_height(i) - self.FLAG_HEIGHT, bg)

            painter.setPen(QPen(QColor("#3E3E3E"), 1))
            painter.drawLine(self.LABEL_WIDTH, row_y, w, row_y)

            painter.setPen(QPen(QColor("#EDEBE9"), 1))
            painter.drawText(8, row_y + self.ROW_HEIGHT // 2 + 4, name)

            lanes, _ = self._row_lanes(blocks)
            for bi in sorted(range(len(blocks)), key=lambda i: (lanes.get(i, 0), i)):
                block = blocks[bi]
                func, start, dur, color_str, *_ = block
                color = QColor(color_str)
                bx = plot_left + int((start / self._total_time) * plot_w)
                is_instant = dur == 0

                if is_instant:
                    fm = painter.fontMetrics()
                    pill_w = max(24, fm.horizontalAdvance(func) + 12)
                    pill_h = self.FLAG_HEIGHT - 2
                    pill_x = max(plot_left, bx - pill_w // 2)
                    pill_y = flag_y + 1

                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    stem_bot = row_y + self.ROW_HEIGHT - 3
                    painter.drawRect(bx - 1, pill_y + pill_h, 2, stem_bot - pill_y - pill_h)
                    painter.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 4, 4)
                    painter.setPen(QPen(QColor("#FFFFFF"), 1))
                    old_font = painter.font()
                    painter.setFont(QFont("Segoe UI", 8))
                    painter.drawText(QRect(pill_x, pill_y, pill_w, pill_h), Qt.AlignCenter, func)
                    painter.setFont(old_font)
                else:
                    by = row_y + 3 + lanes.get(bi, 0) * self.LANE_OFF
                    bh = self.BLOCK_H
                    bw = max(4, int((dur / self._total_time) * plot_w))
                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(bx, by, bw, bh, 4, 4)
                    if bw > 40:
                        painter.setPen(QPen(QColor("#FFFFFF"), 1))
                        painter.drawText(bx + 4, by + bh // 2 + 4, func)

                if self._sel_dev == i and self._sel_block == bi:
                    painter.setBrush(Qt.NoBrush)
                    pen = QPen(QColor("#FFFFFF"), 2)
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    if is_instant:
                        painter.drawRoundedRect(pill_x - 1, pill_y - 1, pill_w + 2, pill_h + 2, 4, 4)
                    else:
                        bw_sel = max(4, int((dur / self._total_time) * plot_w))
                        painter.drawRoundedRect(bx - 1, by - 1, bw_sel + 2, bh + 2, 4, 4)
                        handle_w = 3
                        handle_h = 10
                        handle_y = by + (bh - handle_h) // 2
                        painter.setBrush(QColor(255, 255, 255, 160))
                        painter.setPen(Qt.NoPen)
                        painter.drawRect(bx - 1, handle_y, handle_w, handle_h)
                        painter.drawRect(bx + bw - handle_w + 1, handle_y, handle_w, handle_h)

                if self._active_dev == i and self._active_block == bi:
                    pen = QPen(QColor("#00FF00"), 3)
                    pen.setStyle(Qt.SolidLine)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(pen)
                    if is_instant:
                        painter.drawRoundedRect(pill_x - 2, pill_y - 2, pill_w + 4, pill_h + 4, 5, 5)
                    else:
                        bw_act = max(4, int((dur / self._total_time) * plot_w))
                        painter.drawRoundedRect(bx - 2, by - 2, bw_act + 4, bh + 4, 5, 5)

        # Phase 3: System block full-height bands (overlay across all rows)
        for row in self._devices:
            if row[2] != "__system__":
                continue
            for block in row[1]:
                _, start, dur, color_str = block[0], block[1], block[2], block[3]
                bx = plot_left + int((start / self._total_time) * plot_w)
                if dur == 0:
                    painter.fillRect(bx, rows_top, 2, rows_height, QColor(255, 255, 255, 40))
                else:
                    bw = max(4, int((dur / self._total_time) * plot_w))
                    band = QColor(color_str)
                    band.setAlpha(20)
                    painter.fillRect(bx, rows_top, bw, rows_height, band)
                    painter.setPen(QPen(QColor(color_str).lighter(120), 1))
                    painter.drawLine(bx, rows_top, bx, rows_top + rows_height)
                    painter.drawLine(bx + bw, rows_top, bx + bw, rows_top + rows_height)

        # Phase 4: System row (on top of bands)
        for i, row in enumerate(self._devices):
            name, blocks, device_type = row[0], row[1], row[2]
            if device_type != "__system__":
                continue
            display_name = "System actions"
            flag_y = origins[i]
            row_y = flag_y + self.FLAG_HEIGHT

            painter.fillRect(0, flag_y, w, self.FLAG_HEIGHT, QColor("#2A2A2A"))
            painter.fillRect(0, row_y, w, self.ROW_HEIGHT, QColor("#2A2A2A"))

            painter.setPen(QPen(QColor("#3E3E3E"), 1))
            painter.drawLine(self.LABEL_WIDTH, row_y, w, row_y)

            font = QFont("Segoe UI", 9, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#EDEBE9"), 1))
            painter.drawText(8, row_y + self.ROW_HEIGHT // 2 + 4, display_name)
            painter.setFont(QFont("Segoe UI", 9))

            for bi, block in enumerate(blocks):
                func, start, dur, color_str, *_ = block
                color = QColor(color_str)
                bx = plot_left + int((start / self._total_time) * plot_w)
                is_instant = dur == 0

                if is_instant:
                    fm = painter.fontMetrics()
                    pill_w = max(24, fm.horizontalAdvance(func) + 12)
                    pill_h = self.FLAG_HEIGHT - 2
                    pill_x = max(plot_left, bx - pill_w // 2)
                    pill_y = flag_y + 1

                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    stem_bot = row_y + self.ROW_HEIGHT - 3
                    painter.drawRect(bx - 1, pill_y + pill_h, 2, stem_bot - pill_y - pill_h)
                    painter.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 4, 4)
                    painter.setPen(QPen(QColor("#FFFFFF"), 1))
                    old_font = painter.font()
                    painter.setFont(QFont("Segoe UI", 8))
                    painter.drawText(QRect(pill_x, pill_y, pill_w, pill_h), Qt.AlignCenter, func)
                    painter.setFont(old_font)
                else:
                    by = row_y + 3
                    bh = self.ROW_HEIGHT - 6
                    bw = max(4, int((dur / self._total_time) * plot_w))
                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(bx, by, bw, bh, 4, 4)
                    if bw > 40:
                        painter.setPen(QPen(QColor("#FFFFFF"), 1))
                        painter.drawText(bx + 4, by + bh // 2 + 4, func)

                if self._sel_dev == i and self._sel_block == bi:
                    painter.setBrush(Qt.NoBrush)
                    pen = QPen(QColor("#FFFFFF"), 2)
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    if is_instant:
                        painter.drawRoundedRect(pill_x - 1, pill_y - 1, pill_w + 2, pill_h + 2, 4, 4)
                    else:
                        bw_sel = max(4, int((dur / self._total_time) * plot_w))
                        painter.drawRoundedRect(bx - 1, by - 1, bw_sel + 2, bh + 2, 4, 4)
                        handle_w = 3
                        handle_h = 10
                        handle_y = by + (bh - handle_h) // 2
                        painter.setBrush(QColor(255, 255, 255, 160))
                        painter.setPen(Qt.NoPen)
                        painter.drawRect(bx - 1, handle_y, handle_w, handle_h)
                        painter.drawRect(bx + bw - handle_w + 1, handle_y, handle_w, handle_h)

                if self._active_dev == i and self._active_block == bi:
                    pen = QPen(QColor("#00FF00"), 3)
                    pen.setStyle(Qt.SolidLine)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(pen)
                    if is_instant:
                        painter.drawRoundedRect(pill_x - 2, pill_y - 2, pill_w + 4, pill_h + 4, 5, 5)
                    else:
                        bw_act = max(4, int((dur / self._total_time) * plot_w))
                        painter.drawRoundedRect(bx - 2, by - 2, bw_act + 4, bh + 4, 5, 5)

        # Phase 5: Scrolling cursor during experiment run
        if self._running and self._cursor_time is not None:
            cx = self._x_from_time(self._cursor_time)
            elapsed_rect = QRect(plot_left, rows_top, cx - plot_left, rows_height)
            painter.fillRect(elapsed_rect, QColor(255, 255, 255, 15))
            painter.setPen(QPen(QColor("#FFD700"), 2))
            painter.drawLine(cx, rows_top, cx, rows_top + rows_height)
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QPen(QColor("#FFD700"), 1))
            label = f"{self._cursor_time:.1f}s"
            fm = painter.fontMetrics()
            lw = fm.horizontalAdvance(label)
            bg = QRect(cx - lw // 2 - 2, rows_top - 4, lw + 4, 16)
            painter.fillRect(bg, QColor("#2D2D2D"))
            painter.drawText(cx - lw // 2, rows_top - 4, lw, 16, Qt.AlignCenter, label)
