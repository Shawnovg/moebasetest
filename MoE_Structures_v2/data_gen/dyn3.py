"""DRAFT generator for the next dynamics dataset ("dyn3"): three regimes that each need a different kind of
memory (level A), switched by a coin flip or by a hidden rule (level B).

State: 4 numbers = two "hands" of a clock, each a 2-D vector (handled as one complex number z).

Level A, what happens inside a regime (all scaled to variance ~1 per coordinate):
  0  SHORT  momentum: z[t+1] = a1 z[t] + a2 z[t-1] + noise          -> needs only the last 2 steps
  1  REPLAY replays the past: z[t+1] = 0.92 z[t+1-P] + noise (P=16) -> needs one look exactly P steps back
  2  CHORD  shows a background clock: z = c1 + c2 + noise, where c1, c2 rotate at two speeds and keep
            running even while hidden                                -> needs a long window to find both rhythms
Level B, what decides the switch:
  "coin"   each step: stay with probability p_stay, else jump to another regime (today's rule)
  "timer"  each regime lasts a fixed time (SHORT 10, REPLAY 20, CHORD 16 steps, +-2), then jumps
"""
from __future__ import annotations

import numpy as np

P = 16
NAMES = ["SHORT", "REPLAY", "CHORD"]
# SHORT: complex AR(2) with roots r1, r2 (damped, slowly turning)
R1, R2 = 0.85 * np.exp(0.15j), 0.40 * np.exp(-0.45j)
A1, A2 = R1 + R2, -R1 * R2
# CHORD: two rotation speeds per hand (rad / step)
OMEGA = np.array([[0.25, 0.90], [0.40, 1.30]])
C_DECAY, REPLAY_GAIN = 0.998, 0.924
DUR = {0: 10, 1: 20, 2: 16}


def _ar2_var(sig2):
    # stationary variance of the complex AR(2) driven by complex noise of variance sig2 (by simulation once)
    rng = np.random.default_rng(0)
    z = np.zeros(2, complex); acc = []
    for t in range(20000):
        z = np.array([A1 * z[0] + A2 * z[1] + np.sqrt(sig2 / 2) * (rng.normal() + 1j * rng.normal()), z[0]])
        if t > 1000: acc.append(abs(z[0]) ** 2)
    return np.mean(acc)


def noise_levels(chord_obs=0.15):
    """Noise per regime so every regime has variance ~1 per coordinate (complex variance 2)."""
    s_short = 2.0 / _ar2_var(1.0)                  # complex innovation variance for SHORT
    s_replay = 2.0 * (1 - REPLAY_GAIN ** 2)
    c_var = (2.0 - 2 * chord_obs ** 2) / 2         # each of the 2 components, complex
    s_c = c_var * (1 - C_DECAY ** 2)
    return {"short": s_short, "replay": s_replay, "chord_comp": s_c, "chord_obs": chord_obs}


def generate(n, length=128, rule="coin", p_stay=0.95, seed=0, burn=64):
    rng = np.random.default_rng(seed)
    nz = noise_levels()
    T = burn + length + 1
    z = np.zeros((n, T, 2), complex)               # observed hands
    reg = np.zeros((n, T), int)
    cn = lambda s, size: np.sqrt(s / 2) * (rng.normal(size=size) + 1j * rng.normal(size=size))
    c = cn(2 * (1 - nz["chord_obs"] ** 2) / 2, (n, 2, 2))   # background clock components [seq, hand, comp]
    z[:, :P + 1] = cn(2.0, (n, P + 1, 2))
    r = rng.integers(0, 3, n)
    left = np.array([DUR[k] + rng.integers(-2, 3) for k in r])
    rot = np.exp(1j * OMEGA)[None]
    for t in range(P, T - 1):
        reg[:, t] = r
        c = C_DECAY * rot * c + cn(nz["chord_comp"], c.shape)
        nxt = np.empty((n, 2), complex)
        m = r == 0; nxt[m] = A1 * z[m, t] + A2 * z[m, t - 1] + cn(nz["short"], (m.sum(), 2))
        m = r == 1; nxt[m] = REPLAY_GAIN * z[m, t + 1 - P] + cn(nz["replay"], (m.sum(), 2))
        m = r == 2; nxt[m] = c[m].sum(-1) + cn(nz["chord_obs"] ** 2 * 2, (m.sum(), 2))
        z[:, t + 1] = nxt
        # level B: next regime
        if rule == "coin":
            jump = rng.random(n) > p_stay
        else:
            left -= 1; jump = left <= 0
        new = (r + rng.integers(1, 3, n)) % 3
        r = np.where(jump, new, r)
        if rule == "timer":
            left = np.where(jump, np.array([DUR[k] for k in r]) + rng.integers(-2, 3, n), left)
    zs = z[:, burn:burn + length + 1]
    states = np.stack([zs[..., 0].real, zs[..., 0].imag, zs[..., 1].real, zs[..., 1].imag], -1).astype(np.float32)
    return states, reg[:, burn:burn + length]
