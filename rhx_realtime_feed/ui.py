import sys
from pathlib import Path

# Ensure project root is on sys.path so package imports resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QInputDialog,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QStatusBar,
    QToolButton, QVBoxLayout, QWidget, QMenu,
)

import qdarkstyle
from qdarkstyle.dark.palette import DarkPalette
from rhx_realtime_feed.experiment import ExperimentManager, ExperimentDialog, RunExperimentDialog
from rhx_realtime_feed.experiment.experiment import ExperimentConfig, SequenceStep, _config_to_dict
from rhx_realtime_feed.telemetry_logger import append_telemetry_line, set_telemetry_file
from rhx_realtime_feed import __version__
from rhx_realtime_feed.updater import UpdateCheckThread, UpdateInfo
from rhx_realtime_feed.experiment.experiment_runner import ExperimentRunner
from rhx_realtime_feed.screens.legacy_main_window import LegacyMainWindow
from rhx_realtime_feed.screens._registry import _DEVICE_CLASSES, _SYSTEM_OPERATIONS
from rhx_realtime_feed.screens.timeline import ExperimentTimeline
from rhx_realtime_feed.screens.stage import FluentExpander, LeftSidebar, RightSidebar, MainStage

BG_DARK = "#1E1E1E"
BG_SURFACE = "#252526"
BG_HEADER = "#2D2D2D"
TEXT_PRIMARY = "#EDEBE9"
ACCENT_BLUE = "#0078D4"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroSense Data")
        self.resize(1700, 980)

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
        self._legacy_window = None
        self._experiment_runner = None
        self.main_stage = MainStage()
        root_layout.addWidget(self.main_stage, 1)

        self._wire_behavior()
        self._setup_status_bar()
        self.main_stage.plot_screen.fps_updated.connect(self._fps_status_label.setText)

    def _close_devices(self):
        for inst in getattr(self, '_run_device_instances', []):
            if inst is not None and hasattr(inst, 'close'):
                try:
                    inst.close()
                except Exception as e:
                    print(f"[UI] Error closing device: {e}")
        self._run_device_instances = []

    def closeEvent(self, event):
        if getattr(self, '_experiment_runner', None) is not None and self._experiment_runner.is_running():
            self._experiment_runner.stop()
        self._close_devices()
        self.main_stage.plot_screen.shutdown_workers()
        if self._legacy_window is not None:
            self._legacy_window.close()
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
        self._legacy_ui_action = help_menu.addAction("Legacy UI\u2026")
        help_menu.addSeparator()
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
        self._legacy_ui_action.triggered.connect(self._on_open_legacy_ui)
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
        if not self._current_experiment_path:
            QMessageBox.warning(self, "No Experiment", "Open or create an experiment first.")
            return
        run_data = ExperimentManager.load_run(run_path)
        timeline = self.main_stage.timeline
        timeline.clear_all()
        self.main_stage.plot_screen.clear_all()
        devices = run_data.get("devices", [])
        for d in devices:
            timeline.add_device(name=d.get("name", "Device"), device_type=d.get("device_type", "rhx"))
            self.main_stage.plot_screen.add_device(d.get("name", "Device"), d.get("device_type", "rhx"))
        name_to_idx = {d[0]: i for i, d in enumerate(timeline._devices) if d[2] != "__system__"}
        if not name_to_idx:
            timeline.add_device(name="Default", device_type="rhx")
            self.main_stage.plot_screen.add_device("Default", "rhx")
            name_to_idx = {d[0]: i for i, d in enumerate(timeline._devices) if d[2] != "__system__"}
        current_time = 0.0
        for step in run_data.get("sequence", []):
            duration = step.get("parameters", {}).get("duration_s", 2.0)
            dev_idx = name_to_idx.get(step.get("device_name", ""))
            if dev_idx is None:
                dev_idx = next(iter(name_to_idx.values()))
            timeline.add_block(dev_idx, step.get("action", ""), start=current_time, duration=duration, params=dict(step.get("parameters", {})))
            current_time += duration
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
        self._current_experiment_path = str(dst)
        config = ExperimentManager.load(dst)
        self._populate_timeline_from_config(config)
        self.setWindowTitle(f"NeuroSense Data \u2014 {config.metadata.experiment_name}")
        self.main_stage.left_sidebar.reload_runs(str(dst / "runs"))

    def _on_about(self):
        QMessageBox.about(self, "NeuroSense Data",
            f"NeuroSense Data v{__version__}\n\nReal-time feed for Intan RHX devices.")

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
            if path:
                self._current_experiment_path = path
                config = ExperimentManager.load(path)
                self._populate_timeline_from_config(config)
                self.setWindowTitle(f"NeuroSense Data \u2014 {config.metadata.experiment_name}")
                self.main_stage.left_sidebar.reload_runs(str(Path(path) / "runs"))

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
        self._current_experiment_path = path
        config = ExperimentManager.load(path)
        self._populate_timeline_from_config(config)
        self.setWindowTitle(f"NeuroSense Data \u2014 {config.metadata.experiment_name}")
        self.main_stage.left_sidebar.reload_runs(str(Path(path) / "runs"))

    def _on_experiment_save(self):
        if not self._current_experiment_path:
            QMessageBox.information(self, "No Experiment",
                                    "No experiment is open. Create or open one first.")
            return
        config = ExperimentManager.load(self._current_experiment_path)
        timeline = self.main_stage.timeline
        devs = timeline._devices

        config.devices = [
            {"name": d[0], "device_type": d[2]}
            for d in devs if d[2] != "__system__"
        ]
        config.execution_control.required_devices = list(set(
            d[2] for d in devs if d[2] != "__system__"
        ))

        sequence = []
        step_id = 1
        for dev in devs:
            for block in dev[1]:
                op_name = block[4] if len(block) >= 5 else block[0]
                params = block[5] if len(block) >= 6 else {}
                p = dict(params)
                p["duration_s"] = block[2] * 60.0
                p["_start"] = block[1]
                sequence.append(SequenceStep(
                    step_id=step_id,
                    action=op_name,
                    parameters=p,
                    device_name=dev[0],
                ))
                step_id += 1
        config.sequence = sequence
        ExperimentManager.save(self._current_experiment_path, config)
        QMessageBox.information(self, "Saved", f"Experiment saved to {self._current_experiment_path}")

    def _on_open_legacy_ui(self):
        if not hasattr(self, '_legacy_window') or self._legacy_window is None:
            self._legacy_window = LegacyMainWindow()
        self._legacy_window.show()
        self._legacy_window.raise_()
        self._legacy_window.activateWindow()

    def _on_play_clicked(self):
        if self._experiment_runner is not None:
            self._experiment_runner.resume()
            self.main_stage.btn_play.setEnabled(False)
            self.main_stage.btn_pause.setEnabled(True)
        else:
            self._on_experiment_run()

    def _on_experiment_run(self):
        if not self._current_experiment_path:
            QMessageBox.information(self, "No Experiment",
                                    "Open or create an experiment first.")
            return

        exp_name = Path(self._current_experiment_path).name
        device_groups = []
        for d in self.main_stage.timeline._devices:
            if d[2] == "__system__":
                continue
            device_type = d[2]
            cls = _DEVICE_CLASSES.get(device_type)
            param_defs = cls.get_config_params() if cls else []
            current_config = d[3] if len(d) >= 4 else {}
            device_groups.append({
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
            self._run_device_instances = dialog.device_instances()
            run_name = Path(self._current_run_path).name
            self.setWindowTitle(
                f"NeuroSense Data \u2014 {exp_name} \u2014 Run: {run_name}"
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
                    p.setdefault("duration_s", block[2] * 60.0)
                    sequence.append(SequenceStep(
                        step_id=step_id,
                        action=op_name,
                        parameters=p,
                        device_name="",
                    ))
                    step_id += 1
                continue
            for block in dev[1]:
                op_name = block[4] if len(block) >= 5 else block[0]
                params = block[5] if len(block) >= 6 else {}
                p = dict(params)
                p["block_label"] = block[0]
                p.setdefault("duration_s", block[2] * 60.0)
                sequence.append(SequenceStep(
                    step_id=step_id,
                    action=op_name,
                    parameters=p,
                    device_name=dev[0],
                ))
                step_id += 1
        return sequence

    def _devices_with_instances(self):
        result = []
        for d in self.main_stage.timeline._devices:
            if d[2] == "__system__":
                result.append(d)
                continue
            inst = None
            for runner_inst in self._run_device_instances:
                if hasattr(runner_inst, 'name') and runner_inst.name == d[0]:
                    inst = runner_inst
                    break
                if hasattr(runner_inst, 'device_type') and runner_inst.device_type == d[2]:
                    inst = runner_inst
            result.append(list(d) + [inst])
        return result

    def _start_experiment_sequence(self, exp_name):
        if hasattr(self, '_experiment_runner') and self._experiment_runner is not None:
            if self._experiment_runner.is_running():
                QMessageBox.information(self, "Already Running", "An experiment is already in progress.")
                return

        # ensure every timeline device has a plot tab before the run starts
        for d in self.main_stage.timeline._devices:
            if d[2] == "__system__":
                continue
            name, dev_type = d[0], d[2]
            if name not in self.main_stage.plot_screen._tabs:
                sr = 1000.0 if dev_type == "smu" else 20000.0
                self.main_stage.plot_screen.add_device(name, dev_type, sample_rate=sr)

        timeline_devs = self._devices_with_instances()
        sequence = self._build_sequence_for_runner()

        if not sequence:
            QMessageBox.information(self, "No Steps", "The experiment has no sequence steps to run.")
            return

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

        # disconnect stale button connections
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

        # init telemetry + run.json
        set_telemetry_file(str(Path(self._current_run_path) / "run.log"))
        append_telemetry_line(f"run_start | {exp_name}")
        config = ExperimentManager.load(self._current_experiment_path)
        devices_info = [
            {"name": d[0], "device_type": d[2],
             "config": d[3] if len(d) >= 4 else {},
             "blocks": [
                 {"label": b[0], "action": b[4] if len(b) >= 5 else b[0],
                  "start_min": round(b[1], 1), "duration_min": round(b[2], 1)}
                 for b in d[1]
             ]}
            for d in timeline_devs if d[2] != "__system__"
        ]
        sequence_info = [
            {"step_id": s.step_id, "action": s.action,
             "parameters": dict(s.parameters), "device_name": s.device_name}
            for s in sequence
        ]
        ExperimentManager.init_run(
            self._current_run_path, _config_to_dict(config),
            devices_info, sequence_info,
        )
        if self._current_experiment_path:
            self.main_stage.left_sidebar.reload_runs(
                str(Path(self._current_experiment_path) / "runs")
            )

        self._exp_run_action.setEnabled(False)
        self.statusBar().showMessage(f"Running: {exp_name}")

    def _on_exp_step_started(self, step_index, device_name, action, duration):
        self.statusBar().showMessage(
            f"Step {step_index + 1}: {device_name} \u2192 {action} ({duration:.1f}s)"
        )
        if hasattr(self, 'progress'):
            total = len(self._build_sequence_for_runner())
            self.progress.setMaximum(total)
            self.progress.setValue(step_index)
        self.main_stage.timeline.set_active_step(step_index)

    def _on_exp_step_completed(self, step_index, device_name, action):
        if hasattr(self, 'progress'):
            self.progress.setValue(step_index + 1)

    def _on_exp_finished(self, success, message):
        self._exp_run_action.setEnabled(True)
        self.progress.setValue(0)
        self.main_stage.timeline.clear_active_step()
        status = "success" if success else "failed"
        ExperimentManager.update_run(self._current_run_path, status)
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
        from rhx_realtime_feed.workers.replay_worker import ReplayWorker
        meta_path = Path(run_path) / "run.json"
        if not meta_path.exists():
            meta_path = Path(run_path) / "metadata.json"
            if not meta_path.exists():
                QMessageBox.warning(self, "Replay", f"No run data in {run_path}")
                return
        import json
        meta = json.loads(meta_path.read_text())
        if "devices" in meta:
            dev_info = meta.get("devices", [{}])[0]
            device_type = dev_info.get("device_type", "rhx")
            cfg = dev_info.get("config", {})
            sr = cfg.get("sample_rate", 20000.0)
            nc = cfg.get("num_channels", 1)
        else:
            device_type = meta.get("device_type", "rhx")
            sr = meta.get("sample_rate", 20000.0)
            nc = meta.get("num_channels", 1)
        replay_name = f"Replay: {Path(run_path).name}"
        self.main_stage.plot_screen.add_device(replay_name, device_type, sample_rate=sr, num_channels=nc)
        worker = ReplayWorker(run_path, replay_name, self)
        worker.data_received.connect(self.main_stage.plot_screen.on_data)
        worker.error.connect(lambda msg: self.statusBar().showMessage(f"Replay error: {msg}"))
        worker.finished.connect(lambda: self.statusBar().showMessage("Replay finished"))
        worker.start()
        # ponytail: keep worker alive via attribute; add worker registry if multiple replays needed
        self._replay_worker = worker

    def _on_exp_error(self, device_name, error_message):
        print(f"[UI] Experiment error: {device_name}: {error_message}")
        self.main_stage.btn_play.setEnabled(True)
        self.main_stage.btn_pause.setEnabled(False)
        self.main_stage.btn_stop.setEnabled(False)

    def _on_user_input_requested(self, message):
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

    def _populate_timeline_from_config(self, config: ExperimentConfig):
        timeline = self.main_stage.timeline
        timeline.clear_all()
        self.main_stage.plot_screen.clear_all()

        # restore devices from stored name + type
        if config.devices:
            for d in config.devices:
                timeline.add_device(name=d["name"], device_type=d.get("device_type", "rhx"))
                self.main_stage.plot_screen.add_device(d["name"], d.get("device_type", "rhx"))
        else:
            # legacy: create one device per required_device type
            for dev_type in config.execution_control.required_devices:
                timeline.add_device(name=dev_type, device_type=dev_type)
                self.main_stage.plot_screen.add_device(dev_type, dev_type)

        # ensure system device row exists for system blocks
        if not any(d[2] == "__system__" for d in timeline._devices):
            timeline._devices.append(["__System__", [], "__system__", {}])
            timeline._update_total_time()
            timeline._update_height()

        # map device name → index
        name_to_idx = {d[0]: i for i, d in enumerate(timeline._devices)}

        # ensure at least one non-system device exists
        non_system_idxs = {k: v for k, v in name_to_idx.items()
                           if timeline._devices[v][2] != "__system__"}
        if not non_system_idxs:
            timeline.add_device(name="Default", device_type="rhx")
            self.main_stage.plot_screen.add_device("Default", "rhx")
            name_to_idx = {d[0]: i for i, d in enumerate(timeline._devices)}
            non_system_idxs = {k: v for k, v in name_to_idx.items()
                               if timeline._devices[v][2] != "__system__"}

        sorted_steps = sorted(config.sequence, key=lambda s: s.parameters.get("_start", 0))
        current_time = 0.0
        for step in sorted_steps:
            start = step.parameters.get("_start", current_time)
            duration = step.parameters.get("duration_s", 2.0) / 60.0
            dev_idx = name_to_idx.get(step.device_name) if step.device_name else None
            if dev_idx is None:
                dev_idx = next(iter(non_system_idxs.values()))
            clean_params = {k: v for k, v in step.parameters.items()
                            if k not in ('duration_s', '_start')}
            timeline.add_block(dev_idx, step.action, start=start, duration=duration, params=clean_params)
            if "_start" not in step.parameters:
                current_time += timeline._devices[dev_idx][1][-1][2]


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
