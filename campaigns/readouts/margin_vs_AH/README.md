# A_H of single-frequency neurons (one script of the margin_vs_AH study)

Section 2 of version 2 of the paper states that even the maximum-margin network of Morwani et al. (one hidden layer,
quadratic activation, full grid) has A_H = 2/5: each of its neurons squares the sum of two single-frequency inputs of equal
amplitude, which also produces terms in a - b, 2a and 2b. `fourier_notions_check.py` (from the study's folder, without comments and docstrings)
checks this numerically at p = 31, with the paper's A_H (the share of a layer's centred activation energy in functions of
a + b over all p^2 pairs):

    python3 -I fourier_notions_check.py

Its output (Python 3.11.15, numpy 2.4.4) includes:
- line 3, quadratic: a single-frequency quadratic neuron has A_H 0.400 (shares a + b 0.400, a - b 0.400, a 0.100, b 0.100);
- line 4: a layer of such neurons with the maximum-margin form (16 per frequency, theta_w = theta_u + theta_v) has A_H 0.400,
  and its logits are functions of a + b only (A_H 1.000000) and classify every pair correctly;
- line 3, ReLU: a bias-free ReLU neuron with the same inputs has A_H 0.114; line 5: A_H is unchanged by an orthogonal mixing
  of the neurons.

The study's other files (a comparison of margins with A_H along our runs, and of the two notions of Fourier features) are not
reported in the paper and are not part of this release.
