import glob
from figs import headline, decay_dial, mechanism, controls
from dfigs import width_map
headline(sorted(glob.glob('runs/seed_p31_lw0.015_lv0.2_s*.json')), 'fig_headline_p31.png',
         'p = 31, one hidden layer (ReLU, width 512), cross-entropy, 80% split, five fresh seeds. All layers trained throughout with fixed hyperparameters:\n'
         'hidden layer support-Polar/√rank step 0.03, decay 0.015; head normalized step 0.03, decay 0.2. No rescale, freeze or controller.')
headline(sorted(glob.glob('runs/seed_p23_lw0.01_lv0.2_s*.json')), 'fig_headline_p23.png',
         'p = 23, one hidden layer (ReLU, width 512), cross-entropy, 80% split, five fresh seeds. Same learner as p = 31 with hidden-layer decay 0.01.')
decay_dial([('runs/s1_p31_c24_lw0.005.json', 'λ_W=0.005 (c24 default)'), ('runs/s1_p31_c24_lw0.01.json', 'λ_W=0.01'), ('runs/s2_p31_lw0.015_lv0.2.json', 'λ_W=0.015'),
            ('runs/s1_p31_c24_lw0.02.json', 'λ_W=0.02'), ('runs/s1_p31_c24_lw0.04.json', 'λ_W=0.04')], 'fig_decay_dial_p31.png',
           'The hidden-layer weight decay sets the plateau: p = 31, one hidden layer, same seed and learner, only λ_W changes (head decay 0.2 → ‖V‖ = 5)')
mechanism('runs/seed_p31_lw0.015_lv0.2_s0.json', 'fig_mechanism_p31.png',
          'Inside the plateau (p = 31, seed 0): the logits shrink as ‖W‖ relaxes while the margins grow, so the training CE stays put')
controls([('runs/s2_p31_lw0.015_lv0.2.json', 'primary: everything keeps training'), ('runs/ctl_p31_freezehidden5200.json', 'hidden layer frozen at 5,200'),
          ('runs/ctl_p31_freezehead5200.json', 'head frozen at 5,200'), ('runs/ctl_p31_nodecay.json', 'no weight decay (λ_W = λ_V = 0)')], 'fig_controls_p31.png',
         'Controls on the same initialization and split (p = 31): the test gain needs both layers to keep learning inside the plateau, and the plateau needs weight decay')
width_map([('runs/h1_p31_w128_lw0.015.json', 'width 128'), ('runs/h1_p31_w256_lw0.015.json', 'width 256'), ('runs/s2_p31_lw0.015_lv0.2.json', 'width 512'), ('runs/h1_p31_w1024_lw0.015.json', 'width 1024')],
          'fig_width_1hl_p31.png', 'One hidden layer, p = 31, λ_W = 0.015, λ_V = 0.2, same seed. The memorization bump (steps 3–7k) shrinks with width while the final level does not;\n'
          'the CE stays flat through grokking only once the two meet (width ≥ 512).')
print('ok')
