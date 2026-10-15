import numpy as np
rng = np.random.default_rng(1)
def parts(H, p):
    a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p)
    C = H - H.mean(0)
    def proj(key):
        M = np.zeros((p, H.shape[1])); np.add.at(M, key, C); M /= p
        return M[key] - M[key].mean(0)
    Pp, Pm, Pa, Pb = proj((a + b) % p), proj((a - b) % p), proj(a), proj(b)
    e = lambda X: (X ** 2).sum() / len(a)
    tot = e(C); out = dict(add=e(Pp), sub=e(Pm), a=e(Pa), b=e(Pb), tot=tot)
    out['rest'] = tot - sum(out[k] for k in ('add', 'sub', 'a', 'b')); return out
p = 31; a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); w = 2 * np.pi / p
u, v = rng.normal(size=p), rng.normal(size=p)
U, V = np.fft.fft(u) / p, np.fft.fft(v) / p
h = (u[a] + v[b]) ** 2
P = parts(h[:, None], p)
print('1. random quadratic neuron: E_add %.6f, formula 4*sum|U_r V_r|^2 %.6f, E_(a-b) %.6f, A_H %.3f' %
      (P['add'], 4 * (np.abs(U[1:] * V[1:]) ** 2).sum(), P['sub'], P['add'] / P['tot']))
best = max(parts(((rng.normal(size=(p, 1)) * 0 + rng.normal(size=p)[:, None])[a] + rng.normal(size=p)[b][:, None]) ** 2, p) for _ in range(1))
mx = 0
for _ in range(2000):
    u = rng.normal(size=p) * rng.uniform(0, 1, size=p) ** 4; v = np.roll(u, rng.integers(p)) * rng.choice([-1, 1])
    q = parts(((u[a] + v[b]) ** 2)[:, None], p); mx = max(mx, q['add'] / q['tot'])
print('2. largest A_H over 2000 random quadratic neurons (u and v with matched spectra): %.3f (bound 1/2)' % mx)
for name, f in (('quadratic', lambda z: z ** 2), ('ReLU, no bias', lambda z: np.maximum(z, 0)), ('ReLU, bias -1', lambda z: np.maximum(z - 1, 0)),
                ('ReLU, bias +1', lambda z: np.maximum(z + 1, 0))):
    z = np.cos(3 * w * a + 0.4) + np.cos(3 * w * b + 1.3)
    q = parts(f(z)[:, None], p)
    print('3. single-frequency neuron, %-14s A_H %.3f   shares: a+b %.3f a-b %.3f a %.3f b %.3f rest %.3f' %
          (name, q['add'] / q['tot'], q['add'] / q['tot'], q['sub'] / q['tot'], q['a'] / q['tot'], q['b'] / q['tot'], q['rest'] / q['tot']))
ths = np.arange(4) * np.pi / 2
F = np.zeros((p * p, p)); Hcols = []
for z in range(1, (p - 1) // 2 + 1):
    for tu in ths:
        for tv in ths:
            hh = (np.cos(z * w * a + tu) + np.cos(z * w * b + tv)) ** 2; Hcols.append(hh)
            F += np.outer(hh, np.cos(tu + tv + z * w * np.arange(p)))
qH = parts(np.stack(Hcols, 1), p); qF = parts(F, p)
print('4. hidden layer A_H %.3f; logits A_H %.6f (16 neurons per frequency, phases in {0, pi/2, pi, 3pi/2}, theta_w = theta_u + theta_v)' %
      (qH['add'] / qH['tot'], qF['add'] / qF['tot']))
pred = F.argmax(1); print('   logits classify every pair correctly:', bool((pred == (a + b) % p).all()))
Hs = np.stack(Hcols, 1); Q, _ = np.linalg.qr(rng.normal(size=(Hs.shape[1], Hs.shape[1])))
q1, q2 = parts(Hs, p), parts(Hs @ Q, p)
print('5. A_H before / after an orthogonal mixing of the neurons: %.4f / %.4f' % (q1['add'] / q1['tot'], q2['add'] / q2['tot']))
