"""Signal-processing primitives: bandpass, FFT integration, spectra, spectrogram."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram

from .config import BP_ORDER, INT_FMIN_HP


def bandpass(data, fs, fmin, fmax, order=BP_ORDER):
    """Zero-phase Butterworth bandpass along axis 0."""
    nyq = fs / 2.0
    fmin = max(fmin, 0.01)
    fmax = min(fmax, 0.99 * nyq)
    sos = butter(order, [fmin / nyq, fmax / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=0)


def integrate_fft(velocity, dt, fmin_hp=INT_FMIN_HP):
    """Frequency-domain integration. Divides V(f) by i·2π·f.

    A floor at fmin_hp prevents DC and ultra-low-frequency drift.
    Works on (N_t,) or (N_t, N_s) arrays.
    """
    N = velocity.shape[0]
    f = np.fft.rfftfreq(N, d=dt)
    f_safe = np.where(f < fmin_hp, fmin_hp, f)
    iomega = 1j * 2.0 * np.pi * f_safe
    if velocity.ndim == 2:
        iomega = iomega[:, np.newaxis]
    V = np.fft.rfft(velocity, axis=0)
    return np.fft.irfft(V / iomega, n=N, axis=0).astype(float)


def amplitude_spectrum(data, fs):
    """Single-sided amplitude spectrum.

    data: (N_t,) or (N_t, N_s)
    Returns: (freqs (N_f,), spec (N_f, ...))
    """
    N = data.shape[0]
    spec = np.abs(np.fft.rfft(data, axis=0)) / N
    spec[1:-1] *= 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, spec


def find_dominant_peaks(freqs, spec_1d, n_peaks=4, f_min=1.0, f_max=500.0):
    """Find the top n_peaks dominant frequencies in a 1-D spectrum.

    Searches within [f_min, f_max] using scipy.signal.find_peaks (minimum
    separation 5 Hz). Returns a list of (freq_hz, amp_normalized) sorted
    by amplitude descending, where amplitudes are normalized to the band
    maximum.
    """
    from scipy.signal import find_peaks as _fp
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_m = freqs[mask]
    s_m = np.asarray(spec_1d)[mask]
    s_max = s_m.max() + 1e-30
    s_norm = s_m / s_max

    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    min_dist = max(10, int(5.0 / df))

    peaks, _ = _fp(s_norm, height=0.02, distance=min_dist)
    if len(peaks) == 0:
        idx = np.argsort(s_norm)[::-1][:n_peaks]
        return [(float(f_m[i]), float(s_norm[i])) for i in idx]

    order = np.argsort(s_norm[peaks])[::-1][:n_peaks]
    return [(float(f_m[peaks[k]]), float(s_norm[peaks[k]])) for k in order]


def compute_spectrogram(x, fs, nperseg=None, noverlap=None, window='hann',
                        scaling='spectrum', mode='magnitude'):
    """Wrapper around scipy.signal.spectrogram with sweep-friendly defaults.

    For the 1-500 Hz / ~10 s log-sweep at fs=5 kHz, the defaults nperseg=fs/2
    and noverlap=0.9*nperseg give ~2 Hz frequency resolution with very
    overlapped time bins so the sweep ridge looks smooth.

    Returns:
        f : (N_f,)  frequency bins [Hz]
        t : (N_t,)  segment centre times [s]
        Sxx : (N_f, N_t)  spectrogram values (magnitude by default)
    """
    x = np.asarray(x).ravel()
    if nperseg is None:
        nperseg = int(fs // 2)
    nperseg = max(8, min(nperseg, len(x)))
    if noverlap is None:
        noverlap = int(0.9 * nperseg)
    noverlap = max(0, min(noverlap, nperseg - 1))
    f, t, Sxx = spectrogram(
        x, fs=fs,
        nperseg=nperseg, noverlap=noverlap,
        window=window, scaling=scaling, mode=mode,
    )
    return f, t, Sxx


def compute_fk_spectrum(data, fs, x):
    """Frequency-wavenumber (F-K) spectrum via 2-D FFT.

    Interpolates sensors onto a uniform spatial grid (minimum sensor spacing),
    applies a Hann window in both dimensions, then rfft along time and
    fft + fftshift along space.

    Parameters
    ----------
    data : (N_t, N_s) array
    fs   : sampling frequency [Hz]
    x    : (N_s,) sensor positions [m]  (need not be uniformly spaced)

    Returns
    -------
    freqs  : (N_f,)     positive frequencies [Hz]
    k      : (N_k,)     wavenumbers [1/m], zero-centred
    FK_db  : (N_f, N_k) power [dB relative to peak]
    dx     : uniform spatial step used [m]
    """
    from scipy.interpolate import interp1d

    data = np.asarray(data, dtype=float)
    x = np.asarray(x, dtype=float)

    dx = float(np.min(np.diff(np.sort(x))))
    x_uniform = np.arange(x.min(), x.max() + dx * 0.5, dx)
    N_x = len(x_uniform)

    f_interp = interp1d(x, data, axis=1, kind='linear',
                        bounds_error=False, fill_value=0.0)
    data_u = f_interp(x_uniform)

    win_t = np.hanning(data_u.shape[0])
    win_x = np.hanning(N_x)
    data_u = data_u * win_t[:, np.newaxis] * win_x[np.newaxis, :]

    FK = np.fft.rfft(data_u, axis=0)
    FK = np.fft.fft(FK, axis=1)
    FK = np.fft.fftshift(FK, axes=1)

    power = np.abs(FK) ** 2
    FK_db = 10.0 * np.log10(power / (power.max() + 1e-30) + 1e-30)

    freqs = np.fft.rfftfreq(data_u.shape[0], d=1.0 / fs)
    k = np.fft.fftshift(np.fft.fftfreq(N_x, d=dx))

    return freqs, k, FK_db, dx
