"""
MarkerManager: Manages marker storage, persistence, and command processing.

Encapsulates all marker-related logic including CRUD, CSV persistence,
and atomic marker snapshots.
"""
import csv
from pathlib import Path
from PyQt5 import QtCore


class MarkerManager:
    """Handles marker storage, persistence, and atomic snapshots."""
    
    def __init__(self):
        """Initialize the marker manager."""
        self._markers = []
        self._pending_markers = []
        self._next_marker_id = 1
        self._marker_cmd_queue = []
        self._marker_rewrite_pending = False
        self._io_mutex = QtCore.QMutex()
        self.markers_csv_path = None
    
    def initialize(self, project_run_dir):
        """
        Initialize marker paths for a new recording session.
        
        Args:
            project_run_dir: Path to the project run directory.
        """
        with QtCore.QMutexLocker(self._io_mutex):
            self.markers_csv_path = Path(project_run_dir) / "markers.csv"
            self._markers = []
            self._pending_markers = []
            self._next_marker_id = 1
            self._marker_cmd_queue = []
            self._marker_rewrite_pending = False
            self._write_markers_csv_locked()
    
    def add_marker(self, sample_index, sample_rate, marker_name=""):
        """
        Add a new marker at the given sample index.
        
        Args:
            sample_index: Sample index in the recording.
            sample_rate: Sample rate (for timestamp calculation).
            marker_name: Optional marker name.
        
        Returns:
            Tuple of (marker_id, timestamp_s, markers_snapshot).
        """
        with QtCore.QMutexLocker(self._io_mutex):
            marker_id = int(self._next_marker_id)
            self._next_marker_id += 1
            
            safe_name = str(marker_name).strip() or f"Marker {marker_id}"
            marker_record = {
                "id": marker_id,
                "sample_index": int(sample_index),
                "timestamp_s": int(sample_index) / float(sample_rate),
                "name": safe_name,
            }
            
            self._markers.append(dict(marker_record))
            self._pending_markers.append(dict(marker_record))
            
            self._write_markers_csv_locked()
            markers_snapshot = self._copy_markers_locked()
        
        return marker_id, marker_record["timestamp_s"], markers_snapshot
    
    def request_rename(self, marker_id, new_name):
        """Queue a rename operation."""
        with QtCore.QMutexLocker(self._io_mutex):
            clean_name = str(new_name).strip()
            if not clean_name:
                return False
            self._marker_cmd_queue.append(("rename", int(marker_id), clean_name))
            return True
    
    def request_delete(self, marker_id):
        """Queue a delete operation."""
        with QtCore.QMutexLocker(self._io_mutex):
            self._marker_cmd_queue.append(("delete", int(marker_id), ""))
            return True
    
    def process_commands(self):
        """
        Process all queued marker commands (rename, delete).
        
        Returns:
            Tuple of (markers_snapshot, csv_path) if changes were made, else (None, None).
        """
        with QtCore.QMutexLocker(self._io_mutex):
            if not self._marker_cmd_queue:
                return None, None
            
            changed = False
            for cmd in self._marker_cmd_queue:
                op = cmd[0]
                marker_id = int(cmd[1])
                
                if op == "rename":
                    new_name = str(cmd[2]).strip()
                    if not new_name:
                        continue
                    updated = False
                    for m in self._markers:
                        if int(m.get("id", -1)) == marker_id:
                            m["name"] = new_name
                            updated = True
                    if updated:
                        for m in self._pending_markers:
                            if int(m.get("id", -1)) == marker_id:
                                m["name"] = new_name
                        changed = True
                
                elif op == "delete":
                    before = len(self._markers)
                    self._markers = [m for m in self._markers if int(m.get("id", -1)) != marker_id]
                    self._pending_markers = [m for m in self._pending_markers if int(m.get("id", -1)) != marker_id]
                    if len(self._markers) != before:
                        changed = True
            
            self._marker_cmd_queue.clear()
            
            if changed:
                self._marker_rewrite_pending = True
                return self._copy_markers_locked(), self.markers_csv_path
            
            return None, None
    
    def get_pending_markers(self):
        """Get current pending markers and clear the queue."""
        with QtCore.QMutexLocker(self._io_mutex):
            pending = [dict(m) for m in self._pending_markers]
            self._pending_markers = []
            return pending
    
    def get_markers(self):
        """Get snapshot of all markers (thread-safe)."""
        with QtCore.QMutexLocker(self._io_mutex):
            return self._copy_markers_locked()
    
    def finalize_session(self):
        """
        Finalize the session: rewrite marker fields in chunks if needed, then close.
        
        Returns:
            Tuple of (markers_snapshot, csv_path).
        """
        with QtCore.QMutexLocker(self._io_mutex):
            if self._marker_rewrite_pending:
                # Note: chunk rewrite logic would go here if we kept chunk references
                # For now, just write final CSV
                self._marker_rewrite_pending = False
            
            self._write_markers_csv_locked()
            return self._copy_markers_locked(), self.markers_csv_path
    
    def _copy_markers_locked(self):
        """Return a copy of all markers (must hold _io_mutex)."""
        return [dict(m) for m in self._markers]
    
    def _write_markers_csv_locked(self):
        """Write markers to CSV (must hold _io_mutex)."""
        if self.markers_csv_path is None:
            return
        with open(self.markers_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "timestamp_s", "name"])
            for m in self._markers:
                w.writerow([
                    int(m.get("id", 0)),
                    f"{float(m.get('timestamp_s', 0.0)):.6f}",
                    str(m.get("name", "")),
                ])
