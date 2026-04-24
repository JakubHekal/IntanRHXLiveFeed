"""
State manager singleton for centralized state access and transition requests.

Provides a thread-safe API for querying and transitioning states from any component.
"""

from typing import Optional
import threading
from PyQt5 import QtCore
from rhx_realtime_feed.state_machine import AppStateMachine, AppState


class StateManager:
    """
    Thread-safe singleton state manager.
    
    Provides centralized access to the state machine and state transition requests.
    Use StateManager.get_instance() to access globally.
    """
    
    _instance: Optional['StateManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize the state manager with a new state machine."""
        self._state_machine = AppStateMachine()
        self._transition_queue = []
        self._pending_transitions = {}
    
    @classmethod
    def get_instance(cls) -> 'StateManager':
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton (for testing purposes)."""
        with cls._lock:
            cls._instance = None
    
    # ─── State Query Methods ───────────────────────────────────────────────────
    
    def get_current_state(self) -> AppState:
        """Get the current application state."""
        return self._state_machine.get_current_state()
    
    def get_current_state_str(self) -> str:
        """Get the current state as a string."""
        return self._state_machine.get_current_state_str()
    
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._state_machine.is_connected()
    
    def is_streaming(self) -> bool:
        """Check if device is streaming or waiting for data."""
        return self._state_machine.is_streaming()
    
    def is_streaming_active(self) -> bool:
        """Check if device is actively streaming (not paused)."""
        return self._state_machine.is_streaming_active()
    
    def is_idle(self) -> bool:
        """Check if in IDLE state."""
        return self.get_current_state() == AppState.IDLE
    
    def is_connecting(self) -> bool:
        """Check if in CONNECTING state."""
        return self.get_current_state() == AppState.CONNECTING
    
    def is_paused(self) -> bool:
        """Check if in PAUSED state."""
        return self.get_current_state() == AppState.PAUSED
    
    def is_connection_lost(self) -> bool:
        """Check if in CONNECTION_LOST state."""
        return self.get_current_state() == AppState.CONNECTION_LOST
    
    def can_transition_to(self, target_state: AppState) -> bool:
        """Check if a transition to target_state is valid."""
        return self._state_machine.can_transition_to(target_state)
    
    # ─── State Transition Methods ──────────────────────────────────────────────
    
    def request_connect(self) -> bool:
        """Request transition to CONNECTING state."""
        return self._do_transition('request_connect', AppState.CONNECTING)
    
    def connection_succeeded(self) -> bool:
        """Signal successful connection, transition to CONNECTED."""
        return self._do_transition('connection_succeeded', AppState.CONNECTED)
    
    def connection_failed(self) -> bool:
        """Signal failed connection, transition back to IDLE."""
        return self._do_transition('connection_failed', AppState.IDLE)
    
    def request_stream(self) -> bool:
        """Request to start streaming, transition to STREAMING."""
        return self._do_transition('request_stream', AppState.STREAMING)
    
    def no_data_available(self) -> bool:
        """Signal no data available, transition to WAITING_FOR_DATA."""
        return self._do_transition('no_data_available', AppState.WAITING_FOR_DATA)
    
    def data_arrived(self) -> bool:
        """Signal data arrived, transition back to STREAMING."""
        return self._do_transition('data_arrived', AppState.STREAMING)
    
    def user_pause(self) -> bool:
        """User requested pause, transition to PAUSED."""
        return self._do_transition('user_pause', AppState.PAUSED)
    
    def user_resume(self) -> bool:
        """User requested resume, transition back to STREAMING."""
        return self._do_transition('user_resume', AppState.STREAMING)
    
    def device_disconnected(self) -> bool:
        """Device disconnected unexpectedly, transition to CONNECTION_LOST."""
        return self._do_transition('device_disconnected', AppState.CONNECTION_LOST)
    
    def request_disconnect(self) -> bool:
        """Request disconnect, transition to DISCONNECTING."""
        return self._do_transition('request_disconnect', AppState.DISCONNECTING)
    
    def disconnect_complete(self) -> bool:
        """Disconnect completed, transition to IDLE."""
        return self._do_transition('disconnect_complete', AppState.IDLE)
    
    # ─── Internal Transition Logic ────────────────────────────────────────────
    
    def _do_transition(self, trigger: str, expected_state: AppState) -> bool:
        """
        Execute a state transition.
        
        Args:
            trigger: The transition trigger name (e.g., 'request_connect')
            expected_state: The expected state after transition
        
        Returns:
            True if transition succeeded, False otherwise
        """
        try:
            trigger_method = getattr(self._state_machine, trigger, None)
            if trigger_method is None:
                print(f"[STATE] Invalid trigger: {trigger}")
                return False

            before_state = self._state_machine.get_current_state()

            may_trigger_method = getattr(self._state_machine, f"may_{trigger}", None)
            if callable(may_trigger_method) and not may_trigger_method():
                if before_state == expected_state:
                    return True
                print(f"[STATE] Transition {trigger} ignored - current={before_state.value}, expected={expected_state.value}")
                return False
            
            # Execute the transition
            trigger_method()
            after_state = self._state_machine.get_current_state()
            
            # Verify we're in the expected state
            if after_state != expected_state:
                print(f"[STATE] Transition {trigger} failed - not in {expected_state.value}")
                return False

            if after_state == before_state:
                return True
            
            print(f"[STATE] Transitioned to {expected_state.value}")
            return True
            
        except Exception as e:
            print(f"[STATE] Transition error: {trigger} -> {e}")
            return False
    
    # ─── Signal Access ────────────────────────────────────────────────────────
    
    def get_state_machine(self) -> AppStateMachine:
        """Get the underlying state machine for signal connections."""
        return self._state_machine
    
    def get_state_changed_signal(self) -> QtCore.pyqtSignal:
        """Get the state_changed PyQt signal."""
        return self._state_machine.state_changed
    
    def get_state_changed_str_signal(self) -> QtCore.pyqtSignal:
        """Get the state_changed_str PyQt signal for string state."""
        return self._state_machine.state_changed_str
    
    def get_state_entering_signal(self) -> QtCore.pyqtSignal:
        """Get the state_entering PyQt signal."""
        return self._state_machine.state_entering
    
    def get_state_exiting_signal(self) -> QtCore.pyqtSignal:
        """Get the state_exiting PyQt signal."""
        return self._state_machine.state_exiting
    
    # ─── Debugging ────────────────────────────────────────────────────────────
    
    def print_state(self):
        """Print current state to console."""
        print(f"[STATE] Current state: {self.get_current_state_str()}")
    
    def get_transition_history(self):
        """Get the history of recent state transitions."""
        return self._state_machine.get_transition_history()
    
    def print_history(self):
        """Print the recent transition history."""
        history = self.get_transition_history()
        print("[STATE] Recent transitions:")
        for entry in history:
            print(f"  {entry['timestamp']}: -> {entry['new_state']}")
