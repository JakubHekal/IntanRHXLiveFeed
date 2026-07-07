from PyQt5.QtCore import pyqtSignal, Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics
from PyQt5.QtWidgets import QWidget, QMenu, QInputDialog

from ._registry import _DEVICE_CLASSES, _SYSTEM_OPERATIONS


class ExperimentTimeline(QWidget):
    ROW_HEIGHT = 36
    FLAG_HEIGHT = 18
    LABEL_WIDTH = 130
    HEADER_HEIGHT = 28
    RESIZE_THRESHOLD = 6
    SNAP = 0.5
    _BLOCK_COLORS = ["#0078D4", "#2B88D8", "#4BA3E3", "#107C10", "#498205",
                     "#D13438", "#E74856", "#F1707A", "#8764B8", "#B146C2", "#C239B3"]

    block_selected = pyqtSignal(int, int, str, str, float, float, dict, str)
    data_changed = pyqtSignal()
    device_selected = pyqtSignal(int, str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._device_counter = 0
        self._devices = [
            ["Intan RHX (Port A)", [
                ["Configure", 0.0, 0, "#2B88D8", "Configure", {}],
                ["Stream", 1.0, 14.0, "#4BA3E3", "Stream", {}],
                ["Stimulus", 8.0, 3.0, "#D13438", "Stimulus", {"channel": 1, "amplitude": 500.0, "waveform": "biphasic", "frequency": 100.0}],
            ], "rhx", {"host": "127.0.0.1", "command_port": 5000, "data_port": 5001, "num_channels": 128, "buffer_duration_sec": 5.0}],
            ["Intan RHX (Port B)", [
                ["Configure", 0.0, 0, "#2B88D8", "Configure", {}],
                ["Stream", 1.0, 15.0, "#4BA3E3", "Stream", {}],
            ], "rhx", {"host": "127.0.0.1", "command_port": 5000, "data_port": 5001, "num_channels": 64, "buffer_duration_sec": 5.0}],
            ["miniSMU MS01", [
                ["Configure", 0.0, 0, "#E74856", "Configure", {}],
                ["Stimulus", 2.0, 6.0, "#F1707A", "Stimulus", {"voltage": 5.0, "current_limit": 0.1, "duration_s": 1.0}],
                ["Measure", 8.0, 0, "#E74856", "Measure", {}],
            ], "smu", {"connection_type": "usb", "port": "COM3", "host": "192.168.1.1", "tcp_port": 3333, "mode": "FVMI"}],
            ["Simulated Actor", [
                ["Configure", 0.0, 0, "#B146C2", "Configure", {}],
                ["Write", 3.0, 3.0, "#C239B3", "Write", {"channel": 1, "value": 5.0}],
                ["Trigger", 6.0, 0.5, "#8764B8", "Trigger", {"channel": 1}],
            ], "simulated_actor", {"num_outputs": 2}],
        ]
        self._devices.append(["__System__", [], "__system__", {}])
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
        self._update_total_time()
        self._update_height()

    def _update_height(self):
        per_device = self.FLAG_HEIGHT + self.ROW_HEIGHT
        h = self.HEADER_HEIGHT + 6 + len(self._devices) * per_device + 16
        self.setMinimumHeight(h)

    def _row_at(self, my):
        if my < self.HEADER_HEIGHT + 6:
            return None
        per_device = self.FLAG_HEIGHT + self.ROW_HEIGHT
        dev_idx = (my - self.HEADER_HEIGHT - 6) // per_device
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return None
        row_origin = self.HEADER_HEIGHT + 6 + dev_idx * per_device
        if not (row_origin <= my < row_origin + per_device):
            return None
        return dev_idx

    def add_device(self, name=None, device_type=None):
        if not name:
            self._device_counter += 1
            name = f"Device {self._device_counter}"
        if device_type is None:
            device_type = "rhx"
        cls = _DEVICE_CLASSES.get(device_type)
        config = {}
        if cls:
            config = {p.name: p.default for p in cls.get_config_params()}
        self._devices.append([name, [], device_type, config])
        self._update_total_time()
        self._update_height()
        self.data_changed.emit()
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
            if d[2] == "__system__":
                system_entry = [d[0], [], d[2], d[3] if len(d) >= 4 else {}]
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

    def set_active_step(self, step_index):
        count = 0
        for i, row in enumerate(self._devices):
            if row[2] == "__system__":
                continue
            blocks = row[1]
            if step_index < count + len(blocks):
                self._active_dev = i
                self._active_block = step_index - count
                self.update()
                return
            count += len(blocks)
        self._active_dev = None
        self._active_block = None
        self.update()

    def clear_active_step(self):
        self._active_dev = None
        self._active_block = None
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
        ops = _SYSTEM_OPERATIONS if device_type == "__system__" else (getattr(_DEVICE_CLASSES.get(device_type), 'get_operations', lambda: [])())
        op_duration = duration
        for op in ops:
            if op.name == op_name:
                label = op.label
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

    def contextMenuEvent(self, event):
        mx, my = event.x(), event.y()
        menu = QMenu(self)
        dev_idx, block_idx, _ = self._block_at(mx, my)
        row_idx = self._row_at(my)

        def _build_add_block_menu(parent_menu, target_dev):
            sub = QMenu("Add Block", parent_menu)
            device_type = self._devices[target_dev][2]
            ops = _SYSTEM_OPERATIONS if device_type == "__system__" else []
            if not ops:
                cls = _DEVICE_CLASSES.get(device_type)
                ops = cls.get_operations() if cls else []
            actions = {}
            for op in ops:
                a = sub.addAction(op.label)
                actions[a] = op.name
            if not actions:
                a = sub.addAction("Generic Block")
                actions[a] = "New Block"
            return sub, actions

        is_system_row = row_idx is not None and self._devices[row_idx][2] == "__system__"
        is_system_dev = dev_idx is not None and self._devices[dev_idx][2] == "__system__"

        if block_idx is not None:
            a_del = menu.addAction(f"Remove  \u00ab{self._devices[dev_idx][1][block_idx][0]}\u00bb")
            menu.addSeparator()
            add_menu, add_actions = _build_add_block_menu(menu, dev_idx)
            menu.addMenu(add_menu)
            if not is_system_dev:
                a_del_d = menu.addAction(f"Remove  \u00ab{self._devices[dev_idx][0]}\u00bb")
            action = menu.exec_(event.globalPos())
            if action == a_del:
                self.remove_block(dev_idx, block_idx)
            elif action in add_actions:
                self.add_block(dev_idx, add_actions[action])
            elif not is_system_dev and action == a_del_d:
                self.remove_device(dev_idx)
        elif row_idx is not None:
            add_menu, add_actions = _build_add_block_menu(menu, row_idx)
            menu.addMenu(add_menu)
            if not is_system_row:
                a_del_d = menu.addAction(f"Remove  \u00ab{self._devices[row_idx][0]}\u00bb")
            action = menu.exec_(event.globalPos())
            if action in add_actions:
                self.add_block(row_idx, add_actions[action])
            elif not is_system_row and action == a_del_d:
                self.remove_device(row_idx)
        else:
            add_dev = menu.addAction("Add Device")
            sys_menu = QMenu("Add System Block", menu)
            sys_actions = {}
            for op in _SYSTEM_OPERATIONS:
                a = sys_menu.addAction(op.label)
                sys_actions[a] = op.name
            menu.addMenu(sys_menu)
            action = menu.exec_(event.globalPos())
            if action is not None:
                if action == add_dev:
                    types = sorted(_DEVICE_CLASSES.keys())
                    type_str, ok = QInputDialog.getItem(self, "Add Device", "Device type:", types, 0, False)
                    if ok and type_str:
                        cls = _DEVICE_CLASSES[type_str]
                        name, ok2 = QInputDialog.getText(self, "Add Device", "Name:", text=cls.name)
                        if ok2:
                            self.add_device(name.strip() or cls.name, type_str)
                elif action in sys_actions:
                    self.add_block(len(self._devices) - 1, sys_actions[action])

    def _update_total_time(self):
        self._total_time = max(
            (s + d for row in self._devices for _, s, d, *_ in row[1]),
            default=20,
        )
        if self._total_time <= 0:
            self._total_time = 20

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
        per_device = self.FLAG_HEIGHT + self.ROW_HEIGHT
        dev_idx = (my - self.HEADER_HEIGHT - 6) // per_device
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return None, None, None
        blocks = self._devices[dev_idx][1]
        row_origin = self.HEADER_HEIGHT + 6 + dev_idx * per_device
        if not (row_origin <= my < row_origin + per_device):
            return None, None, None
        in_flag = my < row_origin + self.FLAG_HEIGHT
        if in_flag:
            by = row_origin
            bh = self.FLAG_HEIGHT
            fm = QFontMetrics(QFont("Segoe UI", 8))
        else:
            by = row_origin + self.FLAG_HEIGHT + 3
            bh = self.ROW_HEIGHT - 6
        for bi in range(len(blocks) - 1, -1, -1):
            func, start, dur, *_ = blocks[bi]
            is_instant = dur == 0
            if in_flag != is_instant:
                continue
            bx = self._x_from_time(start)
            if is_instant:
                pill_w = max(24, fm.horizontalAdvance(func) + 12)
                pill_x = max(self._plot_left(), bx - pill_w // 2)
                if pill_x <= mx <= pill_x + pill_w and by <= my < by + bh:
                    return dev_idx, bi, "body"
            else:
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

        num_ticks = max(2, int(self._total_time / 2))
        for i in range(num_ticks + 1):
            x = plot_left + int((i / num_ticks) * plot_w)
            painter.drawLine(x, self.HEADER_HEIGHT, x, self.HEADER_HEIGHT + 4)
            painter.drawText(x - 10, self.HEADER_HEIGHT + 16, str(i * 2))

        per_device = self.FLAG_HEIGHT + self.ROW_HEIGHT
        rows_top = self.HEADER_HEIGHT + 6
        rows_height = len(self._devices) * per_device

        # Phase 2: Draw non-system device rows
        for i, row in enumerate(self._devices):
            name, blocks, device_type = row[0], row[1], row[2]
            if device_type == "__system__":
                continue
            flag_y = rows_top + i * per_device
            row_y = flag_y + self.FLAG_HEIGHT

            bg = QColor("#252526") if i % 2 == 0 else QColor("#1E1E1E")
            painter.fillRect(0, flag_y, w, self.FLAG_HEIGHT, bg)
            painter.fillRect(0, row_y, w, self.ROW_HEIGHT, bg)

            painter.setPen(QPen(QColor("#3E3E3E"), 1))
            painter.drawLine(self.LABEL_WIDTH, row_y, w, row_y)

            painter.setPen(QPen(QColor("#EDEBE9"), 1))
            painter.drawText(8, row_y + self.ROW_HEIGHT // 2 + 4, name)

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
            flag_y = rows_top + i * per_device
            row_y = flag_y + self.FLAG_HEIGHT

            painter.fillRect(0, flag_y, w, self.FLAG_HEIGHT, QColor("#2A2A2A"))
            painter.fillRect(0, row_y, w, self.ROW_HEIGHT, QColor("#2A2A2A"))

            painter.setPen(QPen(QColor("#3E3E3E"), 1))
            painter.drawLine(self.LABEL_WIDTH, row_y, w, row_y)

            font = QFont("Segoe UI", 9, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#EDEBE9"), 1))
            painter.drawText(8, row_y + self.ROW_HEIGHT // 2 + 4, name)
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
