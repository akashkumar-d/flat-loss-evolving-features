# Campaign X figure previews (2026-10-11)

Made before campaign X ran, to check the analysis and the figure design: the transformer panels use one seed per setting
(the campaign-W replays R5 seed 1 and R4 seed 3, rebuilt in campaign X's record format by `testdata/make_testdata.py`, heavy
logs every 200 updates and runs cut at 8000 / 8600 updates); the MLP panels use the eight campaign-F seeds of `mlp_side/`.
Not campaign X results. `analyze_X.py` and `plot_X.py` regenerate the real figures from `collected/results` once the
sessions are done.
