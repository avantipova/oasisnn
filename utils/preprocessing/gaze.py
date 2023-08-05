import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import sosfiltfilt, butter

def preprocess_gaze(
    gaze_path: str,
    info_path: str,
    fmin: float = 0.1,
    fmax: float = 3,
    fs: int = 61,
    f_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analyze gaze gaze and extract relevant information.

    Args:
        gaze_path (str): Path to the gaze data CSV file.
        info_path (str): Path to the info data CSV file.
        fmin (float, optional): Minimum frequency for bandpass filter. Defaults to 0.1.
        fmax (float, optional): Maximum frequency for bandpass filter. Defaults to 3.
        fs (int, optional): Sampling frequency. Defaults to 61.
        f_order (int, optional): Order of the filter. Defaults to 4.

    Returns:
        tuple: A tuple containing time_series and ar_val. ar_val[:, 0] = arousal, ar_val[:, 1] = valence
    """

    logging.debug("Loading gaze and info data...")
    gaze = pd.read_csv(gaze_path)
    info = pd.read_csv(info_path)

    gaze_time_col = [col for col in gaze.columns if col.startswith('TIME(')][0]
    gaze_start = datetime.strptime(gaze_time_col[5:-1], '%Y/%m/%d %H:%M:%S.%f')
    img_start = datetime.strptime(info.start_time_system[0], '%d/%m/%y %H:%M:%S.%f')

    time_difference = abs((img_start.timestamp() - gaze_start.timestamp()) * 1000)
    if time_difference > 1_000_000:
        raise ValueError('Time difference between start of recording and image presentation is too large')

    fs = gaze.CNT.iloc[-1] / gaze[gaze_time_col].iloc[-1]
    counter = (
        np.array(
            info.time_Images[1::2] - ( ( img_start.timestamp() - gaze_start.timestamp() ) * 1000 )
        ) / 1000 * fs
    ).astype(int)

    arousal = np.array(info.form_response[1::2]).astype(int)
    valence = np.array(info.form_response[2::2]).astype(int)

    logging.debug("Interpolating missing data in gaze...")
    to_interpolate = ['FPOGX', 'FPOGY', 'LPMM', 'RPMM', 'GSR', 'HR']
    gaze[to_interpolate] = gaze[to_interpolate].interpolate(method='linear', axis=0)

    logging.debug("Calculating derived columns...")
    gaze['ep_begin'] = np.zeros(len(gaze))
    gaze['ep_begin'][counter] = 1
    gaze = gaze
    gaze['mean_PMM'] = (gaze.LPMM + gaze.RPMM) / 2
    gaze.mean_PMM = gaze.mean_PMM.rolling(window=10, min_periods=0, center=True).mean()
    gaze.GSR = gaze.GSR.rolling(window=100, min_periods=0, center=True).mean()

    logging.debug("Applying bandpass filter to GSR...")
    low = fmin / (fs / 2)
    high = fmax / (fs / 2)
    sos = butter(f_order, [low, high], btype='bandpass', output='sos')
    gaze.GSR = sosfiltfilt(sos, gaze.GSR)
    gaze.HR = gaze.HR.rolling(window=100, min_periods=0, center=True).mean()

    logging.debug("Extracting epoch information...")
    idx = np.where(gaze.ep_begin == 1)[0]
    ns = len(idx)
    t_range = 300
    vars_ = ['FPOGX', 'FPOGY', 'mean_PMM', 'GSR', 'HR']

    time_series = np.zeros((ns, t_range, len(vars_)))
    ar_val = np.zeros((ns, 2))
    ar_val[:, 0] = arousal
    ar_val[:, 1] = valence

    for n, i in enumerate(idx[1:-1]):
        time_series[n] = np.array(gaze[vars_][i:(i + t_range)])

    logging.info(f'NaN count in time_series: {np.isnan(time_series).sum()}')

    return time_series, ar_val