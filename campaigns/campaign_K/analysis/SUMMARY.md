# Campaign K, development stage: results

96 of 96 runs scored with campaign C's per-run endpoint.

## p = 23, depth 3

| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| K_D3_p23_lv0.2322_lw0.015 | 0.2322 | 0.015 | 0/3 | share | 0.21 | 0 | 0.37 | 28160 | 65540 | 1.000 |
| K_D3_p23_lv0.2322_lw0.025 | 0.2322 | 0.025 | 0/3 | share, final; share, final, F02 | 0.45 | 0 | 0.58 | 24240 | 88160 | 0.972 |
| K_D3_p23_lv0.2696_lw0.015 | 0.2696 | 0.015 | 0/3 | share | 0.18 | 0 | 0.44 | 28520 | 73460 | 1.000 |
| K_D3_p23_lv0.2696_lw0.025 | 0.2696 | 0.025 | 0/3 | share, final; share, final, F02 | 0.49 | 0 | 0.64 | 26120 | 92740 | 0.925 |

Selected: none (no development run passes)

## p = 23, depth 4

| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| K_D4_p23_lv0.2322_lw0.015 | 0.2322 | 0.015 | 0/3 | share, F02; share, final, F02 | 0.40 | 0 | 0.41 | 33560 | 89300 | 0.991 |
| K_D4_p23_lv0.2322_lw0.025 | 0.2322 | 0.025 | 0/3 | share; share, final | 0.55 | 0 | 0.61 | 26320 | 103360 | 0.925 |
| K_D4_p23_lv0.2696_lw0.015 | 0.2696 | 0.015 | 0/3 | share, F02; share, final, F02 | 0.37 | 0 | 0.47 | 34000 | 97780 | 0.962 |
| K_D4_p23_lv0.2696_lw0.025 | 0.2696 | 0.025 | 0/3 | share, final | 0.60 | 0 | 0.66 | 26160 | 96800 | 0.887 |

Selected: none (no development run passes)

## p = 47, depth 3

| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| K_D3_p47_lv0.1319_lw0.015 | 0.1319 | 0.015 | 0/3 | share | 0.36 | 0 | 0.54 | 14320 | 20080 | 1.000 |
| K_D3_p47_lv0.1319_lw0.025 | 0.1319 | 0.025 | 0/3 | share | 0.00 | 0 | 0.70 | 11760 | 16440 | 1.000 |
| K_D3_p47_lv0.1624_lw0.015 | 0.1624 | 0.015 | 0/3 | share | 0.59 | 0 | 0.62 | 14380 | 20760 | 1.000 |
| K_D3_p47_lv0.1624_lw0.025 | 0.1624 | 0.025 | 0/3 | share; share, fit | 0.00 | 0 | 0.75 | 11660 | 17120 | 1.000 |

Selected: none (no development run passes)

## p = 47, depth 4

| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| K_D4_p47_lv0.1319_lw0.015 | 0.1319 | 0.015 | 3/3 | pass | 0.85 | 0 | 0.64 | 18980 | 26940 | 1.000 |
| K_D4_p47_lv0.1319_lw0.025 | 0.1319 | 0.025 | 0/3 | fit | 0.86 | 192 | 0.74 | 15840 | 22540 | 1.000 |
| K_D4_p47_lv0.1624_lw0.015 | 0.1624 | 0.015 | 3/3 | pass | 0.95 | 0 | 0.70 | 19120 | 28100 | 1.000 |
| K_D4_p47_lv0.1624_lw0.025 | 0.1624 | 0.025 | 0/3 | fit | 1.00 | 585 | 0.79 | 16040 | 23580 | 1.000 |

Selected: K_D4_p47_lv0.1624_lw0.015 (decided by: min share)

## p = 61, depth 3

| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| K_D3_p61_lv0.1016_lw0.015 | 0.1016 | 0.015 | 0/3 | fit | 1.00 | 208 | 0.58 | 13740 | 17720 | 1.000 |
| K_D3_p61_lv0.1016_lw0.025 | 0.1016 | 0.025 | 0/3 | share | 0.00 | 0 | 0.72 | 10900 | 13900 | 1.000 |
| K_D3_p61_lv0.1426_lw0.015 | 0.1426 | 0.015 | 0/3 | fit | 1.00 | 507 | 0.69 | 13800 | 18680 | 1.000 |
| K_D3_p61_lv0.1426_lw0.025 | 0.1426 | 0.025 | 0/3 | share; share, F02, fit; share, fit | 0.00 | 217 | 0.80 | 11200 | 14920 | 1.000 |

Selected: none (no development run passes)

## p = 61, depth 4

| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| K_D4_p61_lv0.1016_lw0.015 | 0.1016 | 0.015 | 0/3 | fit | 0.91 | 452 | 0.68 | 17820 | 24040 | 1.000 |
| K_D4_p61_lv0.1016_lw0.025 | 0.1016 | 0.025 | 0/3 | fit | 1.00 | 570 | 0.77 | 14740 | 19860 | 1.000 |
| K_D4_p61_lv0.1426_lw0.015 | 0.1426 | 0.015 | 0/3 | fit | 0.94 | 541 | 0.77 | 17880 | 24400 | 1.000 |
| K_D4_p61_lv0.1426_lw0.025 | 0.1426 | 0.025 | 0/3 | fit | 1.00 | 586 | 0.83 | 14420 | 20160 | 1.000 |

Selected: none (no development run passes)

## p = 101, depth 3

| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| K_D3_p101_lv0.0614_lw0.015 | 0.0614 | 0.015 | 0/3 | fit; share, fit | 0.46 | 302 | 0.72 | 13440 | 17220 | 1.000 |
| K_D3_p101_lv0.0614_lw0.025 | 0.0614 | 0.025 | 0/3 | share; share, fit | 0.00 | 388 | 0.77 | 10740 | 13380 | 1.000 |
| K_D3_p101_lv0.1108_lw0.015 | 0.1108 | 0.015 | 0/3 | fit | 1.00 | 667 | 0.83 | 13100 | 17440 | 1.000 |
| K_D3_p101_lv0.1108_lw0.025 | 0.1108 | 0.025 | 0/3 | fit; share, fit | 0.00 | 1071 | 0.88 | 10480 | 13860 | 0.999 |

Selected: none (no development run passes)

## p = 101, depth 4

| setting | lam_v | lam_w | passes | failed checks | share (min) | relapses in band (median) | CE peak / log p | t10 | t90 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| K_D4_p101_lv0.0614_lw0.015 | 0.0614 | 0.015 | 0/3 | share, fit | 0.60 | 441 | 0.79 | 16620 | 21440 | 1.000 |
| K_D4_p101_lv0.0614_lw0.025 | 0.0614 | 0.025 | 0/3 | fit | 1.00 | 507 | 0.83 | 13440 | 16980 | 0.999 |
| K_D4_p101_lv0.1108_lw0.015 | 0.1108 | 0.015 | 0/3 | fit | 0.99 | 698 | 0.87 | 16320 | 23020 | 0.995 |
| K_D4_p101_lv0.1108_lw0.025 | 0.1108 | 0.025 | 0/3 | fit | 1.00 | 2234 | 0.90 | 13280 | 18620 | 0.997 |

Selected: none (no development run passes)

