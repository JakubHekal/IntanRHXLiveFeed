from PyQt5.QtWidgets import QWidget, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QLineEdit, QFormLayout
from rhx_realtime_feed.device.base import ParamDef


def build_param_widget(param_def: ParamDef, current_value=None) -> QWidget:
    value = current_value if current_value is not None else param_def.default
    if param_def.dtype == "float":
        w = QDoubleSpinBox()
        w.setRange(param_def.min_val or -1e6, param_def.max_val or 1e6)
        w.setDecimals(3)
        w.setValue(float(value or 0.0))
        return w
    elif param_def.dtype == "int":
        w = QSpinBox()
        w.setRange(int(param_def.min_val or 0), int(param_def.max_val or 999999))
        w.setValue(int(value or 0))
        return w
    elif param_def.dtype == "bool":
        w = QCheckBox()
        w.setChecked(bool(value))
        return w
    elif param_def.dtype == "choice":
        w = QComboBox()
        w.addItems(param_def.choices or [])
        if value in (param_def.choices or []):
            w.setCurrentText(value)
        return w
    elif param_def.dtype == "channel_list":
        from rhx_realtime_feed.screens.channel_selector import ChannelSelector
        w = ChannelSelector()
        w.setValue(str(value or ""))
        return w
    else:
        w = QLineEdit(str(value or ""))
        return w


def read_param_widget(param_def: ParamDef, widget: QWidget):
    if param_def.dtype == "float":
        return widget.value()
    elif param_def.dtype == "int":
        return widget.value()
    elif param_def.dtype == "bool":
        return widget.isChecked()
    elif param_def.dtype == "choice":
        return widget.currentText()
    elif param_def.dtype == "channel_list":
        return widget.value()
    else:
        return widget.text()


def populate_form_from_params(form_layout: QFormLayout, param_defs: list, current_values: dict, store: list):
    while form_layout.count():
        item = form_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
    store.clear()
    for pd in param_defs:
        val = current_values.get(pd.name, pd.default)
        w = build_param_widget(pd, val)
        form_layout.addRow(pd.label + ":", w)
        store.append((pd.name, w, pd))


def connect_param_signals(store: list, slot):
    from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QLineEdit
    from rhx_realtime_feed.screens.channel_selector import ChannelSelector
    for name, w, pd in store:
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            w.valueChanged.connect(slot)
        elif isinstance(w, QCheckBox):
            w.stateChanged.connect(slot)
        elif isinstance(w, QComboBox):
            w.currentTextChanged.connect(slot)
        elif isinstance(w, QLineEdit):
            w.textChanged.connect(slot)
        elif isinstance(w, ChannelSelector):
            w.valueChanged.connect(slot)


def gather_params(store: list) -> dict:
    return {name: read_param_widget(pd, w) for name, w, pd in store}
