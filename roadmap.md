# RHX Realtime Feed Roadmap

## Scope and Goals
This roadmap covers the core application only:
- main.py
- state_machine.py
- state_manager.py
- workers/
- screens/
- processing/

Primary goal: improve realtime performance first, while fixing reliability issues and reducing technical debt with low-risk incremental changes.

## Guiding Principles
- Keep behavior stable while refactoring internals.
- Prioritize high-impact bottlenecks before architecture cleanups.
- Validate each phase with essential smoke and reliability checks.
- Avoid broad rewrites; prefer small, reversible changes.

## Phase 1: Baseline and Guardrails
### Objectives
- Establish measurable baseline performance and reliability.
- Add minimal observability to compare before/after changes.

### Tasks
1. [In progress] Add lightweight timing and health logging in acquisition, processing, and plotting paths.
  - Implemented periodic telemetry in workers/rhx_worker.py, workers/processing_worker.py, and screens/plot_screen.py.
  - Added persistent run-scoped telemetry logging to `telemetry.txt` inside each recording run directory.
2. [Pending] Record baseline metrics from representative runs:
   - CPU usage
   - UI update cadence
   - data-to-plot latency
   - reconnect time
3. [Pending] Define and run essential smoke checks:
   - connect -> stream -> pause -> resume -> disconnect -> reconnect
   - marker add/edit/delete sanity during streaming

### Exit Criteria
- Baseline metrics captured and reproducible.
- Smoke checklist documented and passing on current behavior.

## Phase 2: High-Impact Performance Fixes
### Objectives
- Reduce latency and CPU overhead in the hottest runtime paths.

### Tasks
1. [In progress] Replace row-by-row CSV writes with buffered batch writes in workers/rhx_worker.py.
  - Implemented initial batching via `writerows()` in the chunk write hot path.
  - Added fast no-marker chunk path using direct `writelines()` to reduce CSV overhead in the common case.
  - Switched to time-based CSV flush policy with larger file buffering to reduce periodic disk I/O stalls.
2. Narrow lock scope around I/O to reduce marker/data-write contention.
  - Moved marker CSV persistence and marker catalog signal emission outside RHX mutex critical sections to shorten lock hold time in hot loop.
3. Convert full-buffer spike/PSD recomputation to incremental updates in workers/processing_worker.py and processing helpers.
  - Implemented waveform extraction optimization to filter only the local segment needed for active spikes (removed full-buffer filtering on each update).
  - Implemented incremental spike histogram cache updates to avoid full-session histogram recomputation on every cycle.
  - Fixed waveform edge-case control flow to avoid skipping worker result emission when segment windows are too short.
4. Reduce repeated ring-buffer linearization on the UI thread in screens/plot_screen.py.
  - Implemented adaptive tail-window scheduling for spike/PSD processing to avoid full ring linear reads and large copies on each processing dispatch.
  - Added sample-clock vs wall-clock drift telemetry in plot diagnostics to detect time-axis undercount during long runs.

### Exit Criteria
- Measurable improvement vs baseline in latency and CPU.
- No behavioral regression in smoke checks.

## Phase 3: Stability and Lifecycle Correctness
### Objectives
- Eliminate race conditions and lifecycle inconsistencies.

### Tasks
1. Make worker stop deterministic (verify thread termination, add bounded retries/failure handling).
  - RHXWorker and ProcessingWorker now expose timeout-based `stop()` returning success/failure.
  - MainWindow now gates connect/disconnect/close flows on confirmed worker shutdown and avoids premature state completion on stop failure.
2. Align state transitions with confirmed worker lifecycle events.
  - Implemented acquisition-state mapping guards in main.py and transition idempotency checks in state_manager.py to remove invalid/duplicate transition noise.
3. Harden connection-loss path to guarantee safe return to idle and clean reconnect.
  - Added automatic connection-lost recovery flow in main.py: transition to disconnecting, deterministic worker stop, UI/session reset, transition to idle, and reconnect prompt.
  - Added resilient async recovery when worker stop times out: force-stop attempt in RHX worker plus background finished-callback cleanup to avoid stuck disconnecting state.
  - Adjusted RHX acquisition cadence to avoid sleeping after successful reads and reduced no-data wait intervals to prevent stream under-read/time-axis drift.
  - Added explicit read pacing to match acquisition window period and immediate repeated-chunk rejection to prevent overlapping-window overcount drift.
  - Updated pacing scheduler to advance on fresh chunks only (retry quickly on stale/no-data reads) to avoid undercount from stale-window slots.
  - Synced worker timing to RHX device-reported sample rate when available so plot and CSV timestamps match hardware cadence.
  - Added short startup all-zero chunk suppression to reduce warm-up zero-window impact on sample-clock drift.
  - **[Latest]** Switched stale-chunk detection from array-equality to buffer-cursor comparison (zero false negatives). Simplified read pacing back to fixed wall-clock period: adaptive measurement caused negative feedback loops when device throughput varied.
4. Ensure marker operations use atomic snapshots during chunk writes.

### Exit Criteria
- Repeated connect/disconnect cycles show no orphan worker threads.
- Forced disconnect scenarios recover predictably.
- Marker consistency preserved under rapid edits.

## Phase 4: Incremental Maintainability Refactor
### Objectives
- Reduce coupling and class complexity without changing external behavior.

### Tasks
1. Decompose RHXWorker internals into focused collaborators:
   - chunk/file writer logic
   - marker storage/serialization logic
2. Extract ring-buffer and processing scheduling responsibilities from PlotScreen.
3. Define explicit data contracts between plotting and processing layers.
4. Centralize runtime config constants with validation at startup.

### Exit Criteria
- Smaller, testable modules with unchanged external interfaces.
- Reduced complexity in RHXWorker and PlotScreen.

## Phase 5: Final Validation and Rollout
### Objectives
- Confirm improvements and close out safely.

### Tasks
1. Re-run baseline scenarios and compare metrics.
2. Execute reconnect stress checks and long-session marker integrity checks.
3. Update README with operational notes and known limits.

### Exit Criteria
- Performance gains validated.
- Reliability checks passing.
- Documentation updated for maintainers.

## Risks and Mitigations
- Risk: hidden behavior changes during refactor.
  - Mitigation: phase-gated smoke checks after each milestone.
- Risk: optimization increases complexity.
  - Mitigation: encapsulate each optimization in focused modules.
- Risk: thread/lifecycle regressions.
  - Mitigation: deterministic stop semantics and repeated lifecycle tests.

## Minimal Validation Checklist
1. Connect and stream with expected UI responsiveness.
2. Pause/resume does not desync state or plots.
3. Save/disconnect leaves no running workers.
4. Reconnect works immediately after disconnect.
5. Marker edits during streaming remain consistent in output files.
