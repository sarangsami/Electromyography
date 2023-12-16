import numpy as np
from scipy.signal import find_peaks


def IEMG(signal):
    return np.sum(np.abs(signal))


def MAV(signal):
    return np.mean(np.abs(signal))


def SSI(signal):
    return np.sum(np.square(signal))


def RMS(signal):
    return np.sqrt(np.mean(np.square(signal)))


def VAR(signal):
    return np.var(signal)


def MYOP(signal):
    return np.sum(np.square(signal)) / len(signal)


def WL(signal):
    return np.sum(np.abs(np.diff(signal)))


def DAMV(signal):
    return np.mean(np.abs(np.diff(signal)))


def M2(signal):
    return np.mean(np.square(np.diff(signal)))


def DVARV(signal):
    return np.var(np.diff(signal))


def DASDV(signal):
    return np.std(np.diff(signal))

# def WAMP(signal, threshold=0):
#     return len(find_peaks(np.abs(signal) >= threshold)[0])


def IASD(signal):
    return np.sum(np.abs(np.diff(signal)))


def IATD(signal):
    return np.sum(np.square(np.diff(signal)))


def IEAV(signal):
    return np.mean(np.abs(np.diff(signal)))


def IALV(signal):
    return np.mean(np.square(np.diff(signal)))


def IE(signal):
    return np.sum(np.square(np.diff(signal))) / len(signal)


def MIN(signal):
    return np.min(signal)


def MAX(signal):
    return np.max(signal)
