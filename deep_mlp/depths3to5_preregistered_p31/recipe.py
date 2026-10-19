import math

K1 = 0.03 * 0.015
SLOPE = 0.9
ROUND = 0.005
R_MIN, R_MAX = 0.10, 1.20
ETA_UNIT = 0.03 * 0.015 / 0.07


def solve_rate(t10, r0, x_target, slope=SLOPE):
    lo, hi = 0.01, 3.0
    f = lambda r: r * K1 * t10 * math.exp(slope * (r - r0)) - x_target
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
    r = round(round(0.5 * (lo + hi) / ROUND) * ROUND, 10)
    return min(max(r, R_MIN), R_MAX)


def first_t10(obs, thr=0.1):
    return next((o['step'] for o in obs if o.get('test_acc', 0) >= thr), None)


def fill_main_cfg(template, rate):
    cfg = dict(template); eta = list(cfg['eta_layers']); eta[-1] = rate * ETA_UNIT; cfg['eta_layers'] = eta
    return cfg


if __name__ == '__main__':
    for t10, want in ((31940, 0.285), (30120, 0.300), (28160, 0.315), (27320, 0.320)):
        r = solve_rate(t10, 0.35, 3.85); print(t10, r, 'T23 used', want); assert abs(r - want) < 1e-9
    print('recipe reproduces T23')
