"""Temporal frequency filtering for raw EEG signals."""
import numpy as np
import mne


def apply_bandpass_filter(
    raw: mne.io.Raw,
    low_freq: float = 8.0,
    high_freq: float = 30.0,
    filter_type: str = "fir",
    verbose: bool = False,
) -> mne.io.Raw:
    """Apply zero-phase bandpass filter to continuous raw EEG data.

    Filtering is applied on continuous raw data prior to epoching to prevent
    edge boundary filtering artifacts.

    Args:
        raw: Preloaded MNE Raw object.
        low_freq: Lower bandpass cutoff frequency in Hz (default 8.0 Hz, mu rhythm).
        high_freq: Upper bandpass cutoff frequency in Hz (default 30.0 Hz, beta rhythm).
        filter_type: 'fir' for FIR filter or 'iir' for Butterworth IIR.
        verbose: MNE verbosity flag.

    Returns:
        Filtered MNE Raw object (copy).
    """
    raw_filtered = raw.copy()
    raw_filtered.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method=filter_type,
        phase="zero",
        fir_design="firwin" if filter_type == "fir" else None,
        verbose=verbose,
    )
    return raw_filtered
