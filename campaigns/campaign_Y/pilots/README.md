# Campaign Y pilots (local, 2026-10-11): seed 1 of R5, 12000 updates

Exploratory runs behind PLAN.md sections 1 and 6, run in the analysis session (2 cores), all on campaign W/X's R5 configuration
with seed 1 (init 7101, split 8101), float64, Fourier readouts every 200 updates, no early stop.

| file | what |
|---|---|
| `runs/W_R5_s1_replay_fourier.json.gz` | no gain: campaign W's R5 seed 1, replayed with the Fourier readouts (campaign X's test data; 8000 updates) |
| `pY_a_etav0.0025_s1.json`, `pY_b_etav0.0012_s1.json`, `runs/pY_*_to7000.json.gz` | slow head (eta_v 0.0025 and 0.0012, tf_runner 1.4), stopped by hand at 7000 updates once the outcome was clear |
| `pG1_g2_tau5k_s1.json`, `pG2_g3_tau4k_s1.json`, `runs/pG*.json.gz` | the gain (gain_init 2, eta_g 2e-4; gain_init 3, eta_g 2.5e-4; lam_g 1), run with the development copy of tf_runner 1.5 (same code, version string 'pilot-G tf_runner 1.5-dev') |
| `analyze_G.py` | crossings, the CE across the held-out rise, the paper's 3% window and W's raw-CE 10% window (a development version of `../analyze_Y.py`) |
| `plot_pilots.py` | `pilots.png` |
