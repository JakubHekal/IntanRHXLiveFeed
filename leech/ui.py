import sys
from copy import deepcopy
from pathlib import Path

# Ensure project root is on sys.path so package imports resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QInputDialog,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QStatusBar,
    QToolButton, QVBoxLayout, QWidget, QMenu,
)

import qdarkstyle
from qdarkstyle.dark.palette import DarkPalette
from leech.experiment import (
    ExperimentManager, ExperimentDialog, RunExperimentDialog, MigrationError,
)
from leech.experiment.experiment import (
    ExperimentConfig, SequenceStep, _config_to_dict,
    SYSTEM_DEVICE_ID, SYSTEM_DEVICE_TYPE,
)
from leech.experiment.migrations import migrate_device_config
from leech.telemetry_logger import append_telemetry_line, set_telemetry_file
from leech import __version__
from leech.updater import UpdateCheckThread, UpdateInfo
from leech.experiment.experiment_runner import ExperimentRunner, sanitize_raw_key
from leech.screens._registry import _DEVICE_CLASSES, _SYSTEM_OPERATIONS
from leech.screens.timeline import ExperimentTimeline
from leech.screens.stage import FluentExpander, LeftSidebar, RightSidebar, MainStage
from leech.plot_settings import save_recent_experiment, load_recent_experiment, load_geometry, save_geometry

BG_DARK = "#1E1E1E"
BG_SURFACE = "#252526"
BG_HEADER = "#2D2D2D"
TEXT_PRIMARY = "#EDEBE9"
ACCENT_BLUE = "#0078D4"


def _prepare_device_record(record):
    if not isinstance(record, dict):
        raise MigrationError("Device record must be an object")
    item = deepcopy(record)
    device_type = item.get("device_type", "unknown")
    device_class = _DEVICE_CLASSES.get(device_type)
    config_version = item.get("config_version", 1)
    if device_class is None:
        current_config = item.get("config", {})
        if not isinstance(current_config, dict):
            raise MigrationError(f"{device_type} config must be an object")
    else:
        current_config, config_version, _ = migrate_device_config(
            device_class, item.get("config", {}), config_version
        )
    item["config"] = current_config
    item["config_version"] = config_version
    return item


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEECH")
        self.resize(1700, 980)
        geo = load_geometry()
        if geo:
            try:
                self.restoreGeometry(geo)
            except TypeError:
                pass

        if getattr(sys, 'frozen', False):
            _icon_path = Path(sys._MEIPASS) / "icon.png"
        else:
            _icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
        if _icon_path.exists():
            self.setWindowIcon(QIcon(str(_icon_path)))

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._create_text_toolbar(root_layout)

        self._current_experiment_path = None
        self._current_run_path = None
        self._run_device_configs = {}
        self._run_device_instances = []
        self._run_device_instance_map = {}
        self._experiment_runner = None
        self.main_stage = MainStage()
        root_layout.addWidget(self.main_stage, 1)

        self._wire_behavior()
        self._prompt_recent_experiment()
        self._setup_status_bar()
        self.main_stage.plot_screen.fps_updated.connect(self._fps_status_label.setText)

    def _read_experiment(self, path):
        try:
            return ExperimentManager.load(path)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self,
                "Experiment Load Failed",
                f"Could not open {path}:\n\n{exc}",
            )
            return None

    def _activate_experiment(self, path):
        config = self._read_experiment(path)
        if config is None:
            return None
        try:
            self._populate_timeline_from_config(config, str(path))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self,
                "Experiment Load Failed",
                f"Could not prepare {path}:\n\n{exc}",
            )
            return None
        self._current_experiment_path = str(path)
        self.setWindowTitle(f"LEECH — {config.metadata.experiment_name}")
        self.main_stage.left_sidebar.reload_runs(str(Path(path) / "runs"))
        return config

    def _read_run(self, path):
        try:
            return ExperimentManager.load_run(path)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self,
                "Run Load Failed",
                f"Could not open {path}:\n\n{exc}",
            )
            return None

    def _prompt_recent_experiment(self):
        path = load_recent_experiment()
        if not path or not Path(path).exists():
            return
        name = Path(path).name
        reply = QMessageBox.question(
            self, "Open Recent Experiment",
            f"Open the last experiment?\n\n{name}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            config_path = Path(path) / "config.json"
            if config_path.exists() and self._activate_experiment(path) is not None:
                save_recent_experiment(str(path))

    def _close_devices(self):
        for inst in getattr(self, '_run_device_instances', []):
            if inst is not None and hasattr(inst, 'close'):
                try:
                    inst.close()
                except Exception as e:
                    print(f"[UI] Error closing device: {e}")
        self._run_device_instances = []
        self._run_device_instance_map = {}

    def _set_edit_mode(self, enabled):
        self.main_stage.set_edit_mode(enabled)
        if enabled:
            self.main_stage.plot_screen.clear_all()
            self.main_stage.plot_screen.set_planning_state()

    def closeEvent(self, event):
        save_geometry(bytes(self.saveGeometry()))
        if getattr(self, '_experiment_runner', None) is not None and self._experiment_runner.is_running():
            self._experiment_runner.stop()
        self._close_devices()
        self.main_stage.plot_screen.shutdown_workers()
        super().closeEvent(event)

    def _create_text_toolbar(self, parent_layout):
        menubar_frame = QFrame()
        menubar_layout = QHBoxLayout(menubar_frame)
        menubar_layout.setContentsMargins(0, 0, 0, 0)
        menubar_layout.setSpacing(0)

        experiment_menu = QMenu("Experiment", self)
        self._exp_new_action = experiment_menu.addAction("New Experiment")
        self._exp_open_action = experiment_menu.addAction("Open Experiment")
        experiment_menu.addSeparator()
        self._exp_save_action = experiment_menu.addAction("Save Experiment")
        self._exp_duplicate_action = experiment_menu.addAction("Duplicate Experiment")
        experiment_menu.addSeparator()
        self._exp_run_action = experiment_menu.addAction("Run Experiment\u2026")
        exp_btn = self._create_menu_button("Experiment", experiment_menu)
        menubar_layout.addWidget(exp_btn)

        help_menu = QMenu("Help", self)
        self._check_update_action = help_menu.addAction("Check for Updates")
        help_menu.addSeparator()
        self._about_action = help_menu.addAction("About")
        help_menu.addAction("Documentation")
        help_btn = self._create_menu_button("Help", help_menu)
        menubar_layout.addWidget(help_btn)

        menubar_layout.addStretch()

        parent_layout.addWidget(menubar_frame)

    def _create_menu_button(self, text: str, menu: QMenu) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setMenu(menu)
        button.setPopupMode(QToolButton.InstantPopup)
        button.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 4px 8px;
                color: #CCCCCC;
                font-size: 11px;
            }
            QToolButton:hover {
                background-color: #3E3E42;
            }
            QToolButton:pressed {
                background-color: #007ACC;
            }
            QToolButton::menu-indicator { image: none; }
        """)
        return button

    def _wire_behavior(self):
        self._exp_new_action.triggered.connect(self._on_experiment_new)
        self._exp_open_action.triggered.connect(self._on_experiment_open)
        self._exp_save_action.triggered.connect(self._on_experiment_save)
        self._exp_duplicate_action.triggered.connect(self._on_experiment_duplicate)
        self._exp_run_action.triggered.connect(self._on_experiment_run)
        self._about_action.triggered.connect(self._on_about)
        self._check_update_action.triggered.connect(self._check_for_updates)
        self.main_stage.left_sidebar.run_selected.connect(self._on_replay_run)
        self.main_stage.left_sidebar.run_action.connect(self._on_run_action)
        self.main_stage.btn_play.clicked.connect(self._on_play_clicked)

    def _setup_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(120)
        status.addPermanentWidget(self.progress)
        self._fps_status_label = QLabel("FPS: 0.0")
        status.addPermanentWidget(self._fps_status_label)

    def _on_run_action(self, action, run_path):
        if action == "rerun":
            self._on_run_rerun(run_path)
        elif action == "rename":
            self._on_run_rename(run_path)
        elif action == "delete":
            self._on_run_delete(run_path)

    def _on_run_rerun(self, run_path):
        if self._experiment_runner is not None and self._experiment_runner.is_running():
            QMessageBox.information(self, "Already Running", "An experiment is already in progress.")
            return
        if not self._current_experiment_path:
            QMessageBox.warning(self, "No Experiment", "Open or create an experiment first.")
            return
        run_data = self._read_run(run_path)
        if run_data is None:
            return
        try:
            devices = [_prepare_device_record(device) for device in run_data.get("devices", [])]
        except (MigrationError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Run Load Failed", str(exc))
            return
        if not devices:
            QMessageBox.information(self, "No Devices", "This run has no devices to rerun.")
            return
        known_ids = {
            device.get("device_id", "").casefold()
            for device in devices
            if device.get("device_id")
        }
        for step in run_data.get("sequence", []):
            device_id = step.get("device_id", "")
            if device_id != SYSTEM_DEVICE_ID and device_id.casefold() not in known_ids:
                QMessageBox.critical(
                    self,
                    "Run Load Failed",
                    f"Run step references unknown device_id {device_id!r}",
                )
                return
        timeline = self.main_stage.timeline
        self._set_edit_mode(True)
        timeline.clear_all()
        self.main_stage.plot_screen.clear_all()
        for device in devices:
            timeline.add_device(
                name=device.get("name", "Device"),
                device_type=device.get("device_type", "unknown"),
                config=device.get("config"),
                device_id=device.get("device_id"),
                config_version=device.get("config_version", 1),
            )
        timeline.add_system_device()
        id_to_idx = {
            row.device_id.casefold(): index
            for index, row in enumerate(timeline._devices)
            if row.device_id
        }
        system_idx = next(
            (index for index, row in enumerate(timeline._devices) if row[2] == SYSTEM_DEVICE_TYPE),
            None,
        )
        current_time = 0.0
        for step in run_data.get("sequence", []):
            params = dict(step.get("parameters", {}))
            duration = params.get("duration_s", 2.0)
            start = params.get("_start", current_time)
            device_id = step.get("device_id", "")
            dev_idx = system_idx if device_id == SYSTEM_DEVICE_ID else id_to_idx.get(device_id.casefold())
            if dev_idx is None:
                raise MigrationError(f"Run step references unknown device_id {device_id!r}")
            clean_params = {
                key: value for key, value in params.items()
                if key not in ("duration_s", "_start", "start_s")
            }
            operation_id = step.get("operation_id", step.get("action", ""))
            device_class = _DEVICE_CLASSES.get(timeline._devices[dev_idx][2])
            if device_class:
                operation_id = device_class.canonical_operation_id(operation_id)
            timeline.add_block(
                dev_idx,
                operation_id,
                start=start,
                duration=duration,
                params=clean_params,
            )
            current_time = max(current_time, start + duration)
        self._on_experiment_run()

    def _on_run_rename(self, run_path):
        old_name = Path(run_path).name
        new_name, ok = QInputDialog.getText(self, "Rename Run", "New name:", text=old_name)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        try:
            ExperimentManager.rename_run(run_path, new_name)
        except FileExistsError:
            QMessageBox.warning(self, "Rename Failed", f"Run '{new_name}' already exists.")
            return
        if self._current_experiment_path:
            self.main_stage.left_sidebar.reload_runs(str(Path(self._current_experiment_path) / "runs"))

    def _on_run_delete(self, run_path):
        name = Path(run_path).name
        confirm = QMessageBox.question(self, "Delete Run", f"Delete run '{name}' permanently?\n\nThis cannot be undone.",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        ExperimentManager.delete_run(run_path)
        if self._current_experiment_path:
            self.main_stage.left_sidebar.reload_runs(str(Path(self._current_experiment_path) / "runs"))

    def _on_experiment_duplicate(self):
        if not self._current_experiment_path:
            QMessageBox.information(self, "No Experiment", "Open or create an experiment first.")
            return
        current_name = Path(self._current_experiment_path).name
        new_name, ok = QInputDialog.getText(self, "Duplicate Experiment", "New experiment name:", text=current_name)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        dst = ExperimentManager.clone_experiment(self._current_experiment_path, new_name)
        if self._activate_experiment(dst) is not None:
            save_recent_experiment(str(dst))

    def _on_about(self):
        QMessageBox.about(self, "LEECH",
            f"LEECH v{__version__}\n\nLive Electrophysiology Experiment Capture Hub.")

    def _check_for_updates(self):
        if getattr(self, '_update_thread', None) is not None and self._update_thread.isRunning():
            QMessageBox.information(self, "Checking", "Update check already in progress.")
            return
        self._update_thread = UpdateCheckThread(__version__, self)
        self._update_thread.result_ready.connect(self._on_update_result)
        self._update_thread.start()

    def _on_update_result(self, result):
        self._update_thread = None
        import json
        if isinstance(result, UpdateInfo):
            if result.available:
                msg = (f"Version {result.latest_version} is available.\n"
                       f"You have {result.current_version}.\n\n"
                       f"Download at:\n{result.release_url}")
                QMessageBox.information(self, "Update Available", msg)
            else:
                QMessageBox.information(self, "Up to Date",
                    f"You have the latest version ({result.current_version}).")
        else:
            QMessageBox.warning(self, "Update Check Failed",
                "Could not check for updates.\n\nCheck your internet connection.")

    def _on_experiment_new(self):
        dialog = ExperimentDialog(self)
        if dialog.exec_():
            path = dialog.result_path()
            if path and self._activate_experiment(path) is not None:
                save_recent_experiment(path)

    def _on_experiment_open(self):
        default_dir = str(self._current_experiment_path) if self._current_experiment_path else str(Path.cwd() / "experiments")
        path = QFileDialog.getExistingDirectory(
            self, "Open Experiment", default_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not path:
            return
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            QMessageBox.warning(self, "Invalid Experiment",
                                "Selected directory does not contain a config.json file.")
            return
        if self._activate_experiment(path) is not None:
            save_recent_experiment(path)

    def _on_experiment_save(self):
        if not self._current_experiment_path:
            QMessageBox.information(self, "No Experiment",
                                    "No experiment is open. Create or open one first.")
            return
        config = self._read_experiment(self._current_experiment_path)
        if config is None:
            return
        timeline = self.main_stage.timeline
        devs = timeline._devices
        existing_devices = {
            device.get("device_id"): device
            for device in config.devices
            if isinstance(device, dict) and device.get("device_id")
        }
        config.devices = []
        for device in devs:
            if device[2] == SYSTEM_DEVICE_TYPE:
                continue
            device_id = getattr(device, "device_id", "")
            item = deepcopy(existing_devices.get(device_id, {}))
            item.update({
                "device_id": device_id,
                "name": device[0],
                "device_type": device[2],
                "config": deepcopy(device[3]) if len(device) >= 4 and isinstance(device[3], dict) else {},
                "config_version": getattr(device, "config_version", 1),
            })
            config.devices.append(item)
        config.execution_control.required_devices = [
            device[2] for device in devs if device[2] != SYSTEM_DEVICE_TYPE
        ]

        step_extras = {}
        for step in config.sequence:
            key = (step.device_id, step.operation_id or step.action, step.parameters.get("block_label", ""))
            step_extras.setdefault(key, []).append(step.extra)
        sequence = []
        step_id = 1
        for device in devs:
            device_id = SYSTEM_DEVICE_ID if device[2] == SYSTEM_DEVICE_TYPE else getattr(device, "device_id", "")
            device_name = "" if device[2] == SYSTEM_DEVICE_TYPE else device[0]
            for block in device[1]:
                operation_id = block[4] if len(block) >= 5 else block[0]
                params = block[5] if len(block) >= 6 else {}
                block_params = dict(params)
                block_params["block_label"] = block[0]
                block_params["duration_s"] = block[2]
                block_params["_start"] = block[1]
                key = (device_id, operation_id, block[0])
                extra = step_extras.get(key, []).pop(0) if step_extras.get(key) else {}
                sequence.append(SequenceStep(
                    step_id=step_id,
                    action=operation_id,
                    operation_id=operation_id,
                    parameters=block_params,
                    device_name=device_name,
                    device_id=device_id,
                    extra=deepcopy(extra),
                ))
                step_id += 1
        config.sequence = sequence
        try:
            ExperimentManager.save(self._current_experiment_path, config)
        except (OSError, MigrationError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Experiment saved to {self._current_experiment_path}")

    def _on_play_clicked(self):
        if self._experiment_runner is not None:
            self._experiment_runner.resume()
            self.main_stage.btn_play.setEnabled(False)
            self.main_stage.btn_pause.setEnabled(True)
        else:
            self._on_experiment_run()

    def _on_experiment_run(self):
        if self._experiment_runner is not None and self._experiment_runner.is_running():
            QMessageBox.information(self, "Already Running", "An experiment is already in progress.")
            return
        if not self._current_experiment_path:
            QMessageBox.information(self, "No Experiment",
                                    "Open or create an experiment first.")
            return

        timeline = self.main_stage.timeline
        if not any(d[2] != SYSTEM_DEVICE_TYPE for d in timeline._devices):
            QMessageBox.information(self, "Add Device", "Add a device before running this experiment.")
            return
        if not self._build_sequence_for_runner():
            QMessageBox.information(self, "No Steps", "Add at least one step before running this experiment.")
            return

        self._set_edit_mode(True)

        exp_name = Path(self._current_experiment_path).name
        device_groups = []
        for d in self.main_stage.timeline._devices:
            if d[2] == SYSTEM_DEVICE_TYPE:
                continue
            device_type = d[2]
            cls = _DEVICE_CLASSES.get(device_type)
            param_defs = cls.get_config_params() if cls else []
            current_config = d[3] if len(d) >= 4 else {}
            device_groups.append({
                "device_id": getattr(d, "device_id", ""),
                "name": d[0],
                "device_type": device_type,
                "device_class": cls,
                "param_defs": param_defs,
                "current_config": current_config,
            })

        dialog = RunExperimentDialog(
            experiment_name=exp_name,
            experiment_path=self._current_experiment_path,
            device_groups=device_groups,
            parent=self,
        )

        if dialog.exec_():
            self._current_run_path = dialog.run_path()
            self._run_device_configs = dialog.device_configs()
            for device in self.main_stage.timeline._devices:
                if device[2] == SYSTEM_DEVICE_TYPE:
                    continue
                device_id = getattr(device, "device_id", "")
                current_config = self._run_device_configs.get(device_id)
                if current_config is None:
                    continue
                merged_config = deepcopy(device[3]) if isinstance(device[3], dict) else {}
                merged_config.update(deepcopy(current_config))
                device[3] = merged_config
                device_class = _DEVICE_CLASSES.get(device[2])
                if device_class:
                    device.config_version = getattr(device_class, "config_version", 1)
            self._run_device_instances, self._run_device_instance_map = dialog.take_devices()
            run_name = Path(self._current_run_path).name
            self.setWindowTitle(
                f"LEECH \u2014 {exp_name} \u2014 Run: {run_name}"
            )

            self._start_experiment_sequence(exp_name)

    def _build_sequence_for_runner(self):
        devs = self.main_stage.timeline._devices
        sequence = []
        step_id = 1
        for dev in devs:
            if dev[2] == "__system__":
                for block in dev[1]:
                    op_name = block[4] if len(block) >= 5 else block[0]
                    params = block[5] if len(block) >= 6 else {}
                    p = dict(params)
                    p["block_label"] = block[0]
                    p.setdefault("duration_s", block[2])
                    p["_start"] = block[1]
                    sequence.append(SequenceStep(
                        step_id=step_id,
                        action=op_name,
                        operation_id=op_name,
                        parameters=p,
                        device_name="",
                        device_id=SYSTEM_DEVICE_ID,
                    ))
                    step_id += 1
                continue
            for block in dev[1]:

                op_name = block[4] if len(block) >= 5 else block[0]
                params = block[5] if len(block) >= 6 else {}
                p = dict(params)
                p["block_label"] = block[0]
                p.setdefault("duration_s", block[2])
                p["_start"] = block[1]
                sequence.append(SequenceStep(
                    step_id=step_id,
                    action=op_name,
                    operation_id=op_name,
                    parameters=p,
                    device_name=dev[0],
                    device_id=getattr(dev, "device_id", ""),
                ))
                step_id += 1
        sequence.sort(key=lambda s: (s.parameters["_start"], s.parameters["duration_s"]))
        for i, s in enumerate(sequence):
            s.step_id = i + 1
        return sequence

    def _devices_with_instances(self):
        result = []
        instance_map = getattr(self, "_run_device_instance_map", {})
        for d in self.main_stage.timeline._devices:
            if d[2] == SYSTEM_DEVICE_TYPE:
                result.append(d)
                continue
            device_id = getattr(d, "device_id", "")
            inst = instance_map.get(device_id)
            if inst is None and not device_id:
                for runner_inst in self._run_device_instances:
                    if hasattr(runner_inst, 'name') and runner_inst.name == d[0]:
                        inst = runner_inst
                        break
            if hasattr(d, "instance"):
                d.instance = inst
                result.append(d)
            else:
                result.append(list(d) + [inst])
        return result

    def _start_experiment_sequence(self, exp_name):
        if hasattr(self, '_experiment_runner') and self._experiment_runner is not None:
            if self._experiment_runner.is_running():
                QMessageBox.information(self, "Already Running", "An experiment is already in progress.")
                return

        timeline_devs = self._devices_with_instances()
        sequence = self._build_sequence_for_runner()
        if not sequence:
            self._close_devices()
            QMessageBox.information(self, "No Steps", "The experiment has no sequence steps to run.")
            return
        config = self._read_experiment(self._current_experiment_path)
        if config is None:
            self._close_devices()
            return
        devices_info = [
            {
                "device_id": getattr(device, "device_id", ""),
                "name": device[0],
                "device_type": device[2],
                "config": deepcopy(device[3]) if len(device) >= 4 and isinstance(device[3], dict) else {},
                "config_version": getattr(device, "config_version", 1),
                "blocks": [
                    {
                        "label": block[0],
                        "action": block[4] if len(block) >= 5 else block[0],
                        "operation_id": block[4] if len(block) >= 5 else block[0],
                        "start_min": round(block[1], 1),
                        "duration_min": round(block[2], 1),
                    }
                    for block in device[1]
                ],
            }
            for device in timeline_devs if device[2] != SYSTEM_DEVICE_TYPE
        ]
        sequence_info = [
            {
                "step_id": step.step_id,
                "action": step.action,
                "operation_id": step.operation_id or step.action,
                "parameters": deepcopy(step.parameters),
                "device_name": step.device_name,
                "device_id": step.device_id,
            }
            for step in sequence
        ]
        try:
            ExperimentManager.init_run(
                self._current_run_path,
                _config_to_dict(config),
                devices_info,
                sequence_info,
            )
        except (OSError, MigrationError, TypeError, ValueError) as exc:
            self._close_devices()
            QMessageBox.critical(self, "Run Initialization Failed", str(exc))
            return

        set_telemetry_file(str(Path(self._current_run_path) / "run.log"))
        append_telemetry_line(f"run_start | {exp_name}")
        self._set_edit_mode(False)
        for device in timeline_devs:
            if device[2] == SYSTEM_DEVICE_TYPE:
                continue
            name, device_type = device[0], device[2]
            if name not in self.main_stage.plot_screen._tabs:
                instance = getattr(device, "instance", None)
                config_data = device[3] if len(device) >= 4 else {}
                sample_rate = instance.sample_rate if instance and instance.sample_rate else (10.0 if device_type == "smu" else 20000.0)
                self.main_stage.plot_screen.add_device(
                    name,
                    device_type,
                    sample_rate=sample_rate,
                    num_channels=config_data.get("num_channels"),
                )
        self.main_stage.plot_screen.set_receiving_state(True)

        self._experiment_runner = ExperimentRunner(
            devices=timeline_devs,
            sequence=sequence,
            run_path=self._current_run_path,
            parent=self,
        )
        self._experiment_runner.step_started.connect(self._on_exp_step_started)
        self._experiment_runner.step_completed.connect(self._on_exp_step_completed)
        self._experiment_runner.experiment_finished.connect(self._on_exp_finished)
        self._experiment_runner.error_occurred.connect(self._on_exp_error)
        self._experiment_runner.data_received.connect(self.main_stage.plot_screen.on_data)
        self._experiment_runner.device_configured.connect(self.main_stage.plot_screen.on_device_configured)
        self._experiment_runner.user_input_requested.connect(self._on_user_input_requested)
        self._experiment_runner.start()

        try:
            self.main_stage.btn_pause.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            self.main_stage.btn_stop.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.main_stage.btn_pause.clicked.connect(self._experiment_runner.pause)
        self.main_stage.btn_pause.clicked.connect(lambda: self.main_stage.btn_pause.setEnabled(False))
        self.main_stage.btn_pause.clicked.connect(lambda: self.main_stage.btn_play.setEnabled(True))
        self.main_stage.btn_stop.clicked.connect(self._experiment_runner.stop)
        self.main_stage.btn_play.setEnabled(False)
        self.main_stage.btn_pause.setEnabled(True)
        self.main_stage.btn_stop.setEnabled(True)
        self.main_stage.timeline.set_running(True)

        if self._current_experiment_path:
            self.main_stage.left_sidebar.reload_runs(
                str(Path(self._current_experiment_path) / "runs")
            )
        self._exp_run_action.setEnabled(False)
        self.statusBar().showMessage(f"Running: {exp_name}")

    def _on_exp_step_started(self, step_index, device_name, action, duration, block_label):
        self.statusBar().showMessage(
            f"Step {step_index + 1}: {device_name} \u2192 {action} ({duration:.1f}s)"
        )
        if hasattr(self, 'progress'):
            total = sum(1 for s in self._build_sequence_for_runner()
                        if s.action not in ("wait_input", "log_event", "pause",
                                             "start_recording", "stop_recording"))
            self.progress.setMaximum(total)
            self.progress.setValue(step_index)
        self.main_stage.timeline.set_active_block(device_name, block_label)

    def _on_exp_step_completed(self, step_index, device_name, action):
        if hasattr(self, 'progress'):
            self.progress.setValue(step_index + 1)

    def _on_exp_finished(self, success, message):
        self._exp_run_action.setEnabled(True)
        self.progress.setValue(0)
        self.main_stage.timeline.set_running(False)
        self.main_stage.timeline.clear_active_step()
        self.main_stage.plot_screen.set_receiving_state(False)
        self.main_stage.plot_screen.clear_all()
        self._set_edit_mode(True)
        status = "success" if success else "failed"
        try:
            ExperimentManager.update_run(self._current_run_path, status)
        except (OSError, MigrationError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Run Status Update Failed", str(exc))
        append_telemetry_line(f"run_end | {status} | {message}")
        if self._current_experiment_path:
            self.main_stage.left_sidebar.reload_runs(
                str(Path(self._current_experiment_path) / "runs")
            )
        if success:
            self.statusBar().showMessage(f"Experiment finished: {message}")
        else:
            self.statusBar().showMessage(f"Experiment aborted: {message}")
        self.main_stage.btn_play.setEnabled(True)
        self.main_stage.btn_pause.setEnabled(False)
        self.main_stage.btn_stop.setEnabled(False)
        self._close_devices()
        self._experiment_runner = None

    def _on_replay_run(self, run_path):
        from leech.workers.replay_worker import ReplayWorker
        metadata = self._read_run(run_path)
        if metadata is None:
            return
        if "devices" in metadata:
            devices = metadata.get("devices", [])
            if not devices:
                QMessageBox.warning(self, "Replay", "Run contains no replayable device.")
                return
            device = devices[0]
            device_id = device.get("device_id", "")
            device_name = device.get("name", "")
            device_type = device.get("device_type", "unknown")
            config = device.get("config", {})
        else:
            device_id = ""
            device_name = metadata.get("name", Path(run_path).name)
            device_type = metadata.get("device_type", "unknown")
            config = metadata
        sample_rate = config.get("sample_rate", 20000.0)
        num_channels = config.get("num_channels", 1)
        raw_root = Path(run_path) / "raw"
        # New format first (name_id), then id-only and name-only legacy layouts.
        candidates = []
        if device_id:
            candidates.append(raw_root / sanitize_raw_key(f"{device_name}_{device_id}"))
            candidates.append(raw_root / sanitize_raw_key(device_id))
        if device_name:
            candidates.append(raw_root / sanitize_raw_key(str(device_name)))
        device_dir = next((c for c in candidates if c.exists()), None)
        if device_dir is None or not device_dir.exists():
            QMessageBox.warning(self, "Replay", f"No raw data found for {device_name or device_type}")
            return
        replay_name = f"Replay: {Path(run_path).name} — {device_name}"
        self.main_stage.plot_screen.add_device(
            replay_name,
            device_type,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        worker = ReplayWorker(
            run_path,
            replay_name,
            self,
            device_dir=device_dir,
            device_id=device_id,
            source_device_name=device_name,
        )
        worker.data_received.connect(self.main_stage.plot_screen.on_data)
        worker.error.connect(lambda msg: self.statusBar().showMessage(f"Replay error: {msg}"))
        worker.finished.connect(lambda: self.statusBar().showMessage("Replay finished"))
        worker.start()
        self._replay_worker = worker

    def _on_exp_error(self, device_name, error_message):
        print(f"[UI] Experiment error: {device_name}: {error_message}")
        self.statusBar().showMessage(f"Experiment error: {device_name}: {error_message}")

    def _on_user_input_requested(self, message):
        self.main_stage.timeline.clear_active_step()
        dialog = QDialog(self)
        dialog.setWindowTitle("User Input Required")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        icon = QLabel("\u26A0\uFE0F")
        icon.setStyleSheet("font-size: 24px;")
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)
        msg = QLabel(message)
        msg.setWordWrap(True)
        msg.setStyleSheet("font-size: 13px;")
        msg.setAlignment(Qt.AlignCenter)
        layout.addWidget(msg)
        btn = QPushButton("OK - Continue")
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE}; color: white;
                border: none; padding: 8px 24px; font-size: 13px;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #106EBE; }}
        """)
        btn.clicked.connect(dialog.accept)
        btn.setDefault(True)
        layout.addWidget(btn, 0, Qt.AlignCenter)
        dialog.exec_()
        if self._experiment_runner is not None and self._experiment_runner._thread is not None:
            self._experiment_runner._thread._input_result = ("ok", True)

    def _populate_timeline_from_config(self, config: ExperimentConfig, experiment_path=None):
        migrated = config.migration_changed
        prepared_devices = []
        if config.devices:
            for device in config.devices:
                prepared = _prepare_device_record(device)
                if (
                    prepared.get("config") != device.get("config")
                    or prepared.get("config_version") != device.get("config_version")
                ):
                    migrated = True
                device.update(prepared)
                prepared_devices.append(prepared)
        else:
            for device_type in config.execution_control.required_devices:
                prepared_devices.append({
                    "name": device_type,
                    "device_type": device_type,
                    "device_id": None,
                    "config": {},
                    "config_version": 1,
                })

        known_ids = {
            device.get("device_id", "").casefold()
            for device in prepared_devices
            if device.get("device_id")
        }
        sorted_steps = sorted(
            config.sequence,
            key=lambda step: step.parameters.get("_start", step.parameters.get("start_s", 0)),
        )
        for step in sorted_steps:
            device_id = step.device_id
            if device_id != SYSTEM_DEVICE_ID and device_id.casefold() not in known_ids:
                raise MigrationError(
                    f"Experiment step references unknown device_id {device_id!r}"
                )

        timeline = self.main_stage.timeline
        timeline.clear_all()
        self.main_stage.plot_screen.clear_all()
        self._set_edit_mode(True)
        for device in prepared_devices:
            timeline.add_device(
                name=device.get("name", "Device"),
                device_type=device.get("device_type", "unknown"),
                config=device.get("config", {}),
                device_id=device.get("device_id"),
                config_version=device.get("config_version", 1),
            )
        timeline.add_system_device()
        id_to_idx = {
            row.device_id.casefold(): index
            for index, row in enumerate(timeline._devices)
            if row.device_id
        }
        system_idx = next(
            (index for index, row in enumerate(timeline._devices) if row[2] == SYSTEM_DEVICE_TYPE),
            None,
        )
        current_time = 0.0
        for step in sorted_steps:
            start = step.parameters.get(
                "_start", step.parameters.get("start_s", current_time)
            )
            duration = step.parameters.get("duration_s", 2.0)
            device_id = step.device_id
            dev_idx = (
                system_idx
                if device_id == SYSTEM_DEVICE_ID
                else id_to_idx.get(device_id.casefold())
            )
            if dev_idx is None:
                raise MigrationError(f"Experiment step references unknown device_id {device_id!r}")
            clean_params = {
                key: value for key, value in step.parameters.items()
                if key not in ("duration_s", "_start", "start_s")
            }
            operation_id = step.operation_id or step.action
            device_class = _DEVICE_CLASSES.get(timeline._devices[dev_idx][2])
            if device_class:
                operation_id = device_class.canonical_operation_id(operation_id)
            timeline.add_block(
                dev_idx,
                operation_id,
                start=start,
                duration=duration,
                params=clean_params,
            )
            current_time = max(current_time, start + duration)

        if config.migration_warnings:
            QMessageBox.warning(
                self,
                "Project Migration Warnings",
                "\n".join(config.migration_warnings),
            )
        target_path = experiment_path or getattr(self, "_current_experiment_path", None)
        if migrated and target_path:
            try:
                ExperimentManager.save(target_path, config, backup=True)
                config.migration_changed = False
            except (OSError, MigrationError, TypeError, ValueError) as exc:
                QMessageBox.warning(
                    self,
                    "Project Migration Failed",
                    f"Could not upgrade project format:\n\n{exc}",
                )


def main():
    import importlib, os

    if '_PYI_SPLASH_IPC' in os.environ and importlib.util.find_spec("pyi_splash"):
        import pyi_splash
        pyi_splash.update_text('UI Loaded ...')
        pyi_splash.close()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    if hasattr(Qt, 'setHighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QApplication(sys.argv)

    if getattr(sys, 'frozen', False):
        icon_path = Path(sys._MEIPASS) / "icon.png"
    else:
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    base_ss = qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=DarkPalette)
    override_ss = f"""
        QMainWindow {{ background-color: {BG_DARK}; }}
        QWidget {{ background-color: {BG_DARK}; }}
        QLabel {{ color: {TEXT_PRIMARY}; background: transparent; }}
        QTreeView {{ background-color: {BG_SURFACE}; color: {TEXT_PRIMARY};
                      border: 1px solid {BG_HEADER}; }}
        QTextEdit {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY};
                     border: 1px solid {BG_HEADER}; }}
        QSplitter::handle {{ background-color: {BG_HEADER}; }}
        QScrollArea {{ background: {BG_DARK}; }}
        QStatusBar {{ background-color: {BG_HEADER}; color: {TEXT_PRIMARY}; }}
        QProgressBar {{ background-color: {BG_SURFACE}; color: {TEXT_PRIMARY};
                        border: 1px solid {BG_HEADER}; text-align: center; }}
        QProgressBar::chunk {{ background-color: {ACCENT_BLUE}; }}
        QTabWidget::pane {{ background-color: {BG_DARK}; border: 1px solid {BG_HEADER}; }}
        QTabBar::tab {{ background-color: {BG_SURFACE}; color: {TEXT_PRIMARY};
                        border: 1px solid {BG_HEADER}; padding: 4px 8px; }}
        QTabBar::tab:selected {{ background-color: {BG_HEADER}; }}
        QComboBox {{ background-color: {BG_SURFACE}; color: {TEXT_PRIMARY};
                     border: 1px solid {BG_HEADER}; padding: 2px 4px; }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY};
                                       selection-background-color: {BG_HEADER}; }}
        QLineEdit, QSpinBox, QDoubleSpinBox {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY};
                                               border: 1px solid {BG_HEADER}; padding: 2px 4px; }}
        QToolBar {{ background-color: {BG_HEADER}; border: none; spacing: 4px; }}
        QToolButton {{ color: {TEXT_PRIMARY}; background: transparent; border: none; padding: 2px 6px; }}
        QToolButton:hover {{ background-color: {BG_SURFACE}; }}
        QToolButton:pressed, QToolButton:checked {{ background-color: {ACCENT_BLUE}; }}
        QMenu {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY}; border: 1px solid {BG_HEADER}; }}
        QMenu::item:selected {{ background-color: {BG_HEADER}; }}
    """
    app.setStyleSheet(base_ss + override_ss)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
