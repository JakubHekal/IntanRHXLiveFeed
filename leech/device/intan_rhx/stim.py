"""
Stimulation parameter helpers for the Intan RHX stim channel registers.

Pulse-per-train is capped at 256 by the RHS chip's 8-bit register field, so
long protocols are delivered as repeated edge-triggered trains: the runner
fires `manualstimtriggerpulse f1` every 256 * pulse_period (one train), and
the chip runs the full train per rising edge. Duration of a timeline block is
the dose: pulse count = trains * 256 = duration / pulse period.

Trigger source must be the software manual trigger (`KeyPressF1`, asserted by
`manualstimtriggerpulse f1`); `DigitalIn01` would wire the trigger to the
connector's physical digital-in pin and the software trigger would be ignored.

`StimParameters::validate()` runs on every per-channel upload and silently
skips the upload on any warning, so this module must never produce an invalid
register map. The clamp/coupling rules below mirror validate():
  * PulseTrainPeriod >= stimDuration (shape-dependent, see below)
  * PostTriggerDelay >= PreStimAmpSettle
  * RefractoryPeriod >= PostStimAmpSettle
"""

STIM_STEP_MICROAMPS = 0.5
MAX_PULSES_PER_TRAIN = 256
MAX_AMPLITUDE_MICROAMPS = 2550.0

# Amplifier channels offer only these three shapes (Monophasic is DAC-only).
SHAPES = ("Biphasic", "BiphasicWithInterphaseDelay", "Triphasic")
POLARITIES = ("NegativeFirst", "PositiveFirst")


def quantize_amplitude(amplitude_uA):
    if amplitude_uA <= 0:
        return 0.0, []
    delivered = int(amplitude_uA / STIM_STEP_MICROAMPS + 0.5) * STIM_STEP_MICROAMPS
    delivered = min(delivered, MAX_AMPLITUDE_MICROAMPS)
    if delivered == 0:
        delivered = STIM_STEP_MICROAMPS
    warnings = []
    if abs(delivered - amplitude_uA) > 1e-9:
        warnings.append(
            f"amplitude {amplitude_uA:g} uA is not a multiple of the "
            f"{STIM_STEP_MICROAMPS:g} uA DAC step; delivering {delivered:g} uA"
        )
    return delivered, warnings


def _stim_duration(shape, phase_duration_us, second_phase_duration_us, interphase_delay_us):
    if shape == "Biphasic":
        return phase_duration_us + second_phase_duration_us
    if shape == "BiphasicWithInterphaseDelay":
        return phase_duration_us + interphase_delay_us + second_phase_duration_us
    return 2.0 * phase_duration_us + second_phase_duration_us  # Triphasic


def build_stim_params(shape="Biphasic", polarity="NegativeFirst",
                      amplitude_uA=0.5, second_amplitude_uA=None,
                      phase_duration_us=100.0, second_phase_duration_us=None,
                      interphase_delay_us=0.0, pulse_period_us=200.0,
                      refractory_period_us=0.0,
                      pre_stim_amp_settle_us=0.0, post_stim_amp_settle_us=0.0):
    warnings = []
    if shape not in SHAPES:
        raise ValueError(f"shape {shape!r} not in {SHAPES}")
    if polarity not in POLARITIES:
        raise ValueError(f"polarity {polarity!r} not in {POLARITIES}")

    if second_amplitude_uA is None:
        second_amplitude_uA = amplitude_uA
    if second_phase_duration_us is None:
        second_phase_duration_us = phase_duration_us

    amp1, w = quantize_amplitude(amplitude_uA)
    warnings.extend(w)
    amp2, w = quantize_amplitude(second_amplitude_uA)
    warnings.extend(w)

    min_period = _stim_duration(shape, phase_duration_us, second_phase_duration_us, interphase_delay_us)
    if pulse_period_us < min_period:
        pulse_period_us = min_period
        warnings.append(
            f"pulse period shorter than the shape's pulse duration ({min_period:g} us); "
            f"clamped to {pulse_period_us:g} us"
        )

    # validate() rule: PostTriggerDelay >= PreStimAmpSettle.
    post_trigger_delay = pre_stim_amp_settle_us
    # validate() rule: RefractoryPeriod >= PostStimAmpSettle.
    if refractory_period_us < post_stim_amp_settle_us:
        refractory_period_us = post_stim_amp_settle_us
        warnings.append(
            f"refractory period raised to {refractory_period_us:g} us "
            f"(must be >= post-stim amp settle {post_stim_amp_settle_us:g} us)"
        )

    params = [
        ("Shape", shape),
        ("Polarity", polarity),
        ("PulseOrTrain", "PulseTrain"),
        ("StimEnabled", "true"),
        ("Source", "KeyPressF1"),
        ("TriggerEdgeOrLevel", "Edge"),
        ("TriggerHighOrLow", "High"),
        ("MaintainAmpSettle", "false"),
        ("EnableAmpSettle", "true"),
        ("EnableChargeRecovery", "false"),
        ("FirstPhaseDurationMicroseconds", phase_duration_us),
        ("SecondPhaseDurationMicroseconds", second_phase_duration_us),
        ("InterphaseDelayMicroseconds", interphase_delay_us),
        ("FirstPhaseAmplitudeMicroAmps", amp1),
        ("SecondPhaseAmplitudeMicroAmps", amp2),
        ("PostTriggerDelayMicroseconds", post_trigger_delay),
        ("PulseTrainPeriodMicroseconds", pulse_period_us),
        ("RefractoryPeriodMicroseconds", refractory_period_us),
        ("PreStimAmpSettleMicroseconds", pre_stim_amp_settle_us),
        ("PostStimAmpSettleMicroseconds", post_stim_amp_settle_us),
        ("PostStimChargeRecovOnMicroseconds", 0.0),
        ("PostStimChargeRecovOffMicroseconds", 0.0),
        ("NumberOfStimPulses", MAX_PULSES_PER_TRAIN),
    ]
    return params, warnings


def _fmt(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _verify():
    """Self-check the edge-trigger train math used by the runner."""
    params, _ = build_stim_params(pulse_period_us=20000.0, refractory_period_us=40000.0)
    p = dict(params)
    assert p["TriggerEdgeOrLevel"] == "Edge", "must edge-trigger for pulse trains"
    assert p["Source"] == "KeyPressF1", "software trigger must pair with manualstimtrigger f1"
    assert p["NumberOfStimPulses"] == MAX_PULSES_PER_TRAIN
    period = p["PulseTrainPeriodMicroseconds"]
    refractory = p["RefractoryPeriodMicroseconds"]
    train_dur = (MAX_PULSES_PER_TRAIN * period + refractory) * 1e-6
    assert abs(train_dur - 5.16) < 1e-9, train_dur
    trains = int(600.0 // train_dur)
    assert trains == 116, trains
    assert trains * MAX_PULSES_PER_TRAIN == 29696, "600 s of 50 Hz = ~30000 pulses"
    print(f"stim self-check OK: {train_dur:.2f}s per train, {trains} trains x "
          f"{MAX_PULSES_PER_TRAIN} = {trains * MAX_PULSES_PER_TRAIN} pulses in 600 s")


if __name__ == "__main__":
    _verify()
