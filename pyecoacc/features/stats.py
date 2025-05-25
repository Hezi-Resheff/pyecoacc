import numpy as np
import scipy

ODBA_LOW_PASS_WINDOW = 10
window = np.ones(ODBA_LOW_PASS_WINDOW) / ODBA_LOW_PASS_WINDOW


def dba(x):
    smooth = np.apply_along_axis(lambda row: np.convolve(row, window, mode='same'), axis=1, arr=x)
    return np.absolute(x - smooth).mean(axis=1)


def count_smooth_peaks(x):
    mat = np.apply_along_axis(lambda row: np.convolve(row, window, mode='same'), axis=1, arr=x)
    left = mat[:, :-2]
    center = mat[:, 1:-1]
    right = mat[:, 2:]

    # Find peaks (center > left AND center > right)
    peaks = (center > left) & (center > right)
    return np.sum(peaks, axis=1)


def count_average_crossings(x):
    row_avgs = np.mean(x, axis=1, keepdims=True)
    above_avg = x > row_avgs

    crossings = np.abs(np.diff(above_avg.astype(int), axis=1))

    return np.sum(crossings, axis=1)


def dominant_freq(x):
    x_centered = x - x.mean(axis=1, keepdims=True)   # Remove DC component
    fft_magnitude = np.abs(np.fft.rfft(x_centered, axis=1))
    dominant_indices = np.argmax(fft_magnitude, axis=1)
    freqs = np.fft.rfftfreq(x.shape[1])
    dominant_freqs = freqs[dominant_indices]
    return dominant_freqs


# The feature registry

single_axis_features = {
    "mean": lambda x: x.mean(axis=1),
    "std": lambda x: x.std(axis=1),
    "skew": lambda x: scipy.stats.skew(x, axis=1),
    "kurt": lambda x: scipy.stats.kurtosis(x, axis=1),
    "min": lambda x: x.min(axis=1),
    "max": lambda x: x.max(axis=1),
    "amplitude": lambda x: x.max(axis=1) - x.min(axis=1),
    "percentile_25": lambda x: np.percentile(x, 25, axis=1),
    "percentile_50": lambda x: np.percentile(x, 50, axis=1),
    "percentile_75": lambda x: np.percentile(x, 75, axis=1),
    "dba": dba,
    "peaks": count_smooth_peaks,
    "average_crossings": count_average_crossings,
    "dominant_frequency": dominant_freq
}

multiple_axis_features = {
    "ODBA": lambda x, y, z, n: dba(x) + dba(y) + dba(z),

    "cov_xy": lambda x, y, z, n: np.array([np.cov(x[i], y[i])[0, 1] for i in range(x.shape[0])]),
    "cov_xz": lambda x, y, z, n: np.array([np.cov(x[i], z[i])[0, 1] for i in range(x.shape[0])]),
    "cov_yz": lambda x, y, z, n: np.array([np.cov(z[i], y[i])[0, 1] for i in range(x.shape[0])]),

    "r_xy": lambda x, y, z, n: np.array([np.corrcoef(x[i], y[i])[0, 1] for i in range(x.shape[0])]),
    "r_xz": lambda x, y, z, n: np.array([np.corrcoef(x[i], z[i])[0, 1] for i in range(x.shape[0])]),
    "r_yz": lambda x, y, z, n: np.array([np.corrcoef(z[i], y[i])[0, 1] for i in range(x.shape[0])]),

    "mean_diff_xy": lambda x, y, z, n: (x - y).mean(axis=1),
    "mean_diff_xz": lambda x, y, z, n: (x - z).mean(axis=1),
    "mean_diff_yz": lambda x, y, z, n: (z - y).mean(axis=1),

    "std_diff_xy": lambda x, y, z, n: (x - y).std(axis=1),
    "std_diff_xz": lambda x, y, z, n: (x - z).std(axis=1),
    "std_diff_yz": lambda x, y, z, n: (z - y).std(axis=1),
}

