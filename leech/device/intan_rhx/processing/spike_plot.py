import numpy as np

PRE_MS  = 0.6
POST_MS = 1.0


def extract_waveforms(xhp, idx, fs, pre_ms=PRE_MS, post_ms=POST_MS):
    pre  = int(round(pre_ms*1e-3*fs))
    post = int(round(post_ms*1e-3*fs))
    valid = idx[(idx - pre >= 0) & (idx + post < len(xhp))]
    if valid.size == 0:
        return valid, np.zeros((0, pre+post))
    idx = valid[:, np.newaxis] + np.arange(-pre, post)
    W = xhp[idx]
    return valid, W
