#!/usr/bin/env python3
import sys
PF, OUT = sys.argv[1], sys.argv[2]
HEIGHT = float(sys.argv[3]) if len(sys.argv) > 3 else 3.7
sys.path.insert(0, PF)
import scheme2 as S2
from matplotlib.ticker import FormatStrFormatter

STEM = 'locked_cohorts_1hl'
_draw2 = S2.draw2


def draw2(cols, out, stem, *a, **k):
    if stem == STEM:
        k['height'] = HEIGHT
        k['align_titles'] = False
    return _draw2(cols, out, stem, *a, **k)


S2.draw2 = draw2
_save = S2.save


def plain_decimals(ax):
    lo, hi = sorted(ax.get_ylim())
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    scale = max(abs(t) for t in ticks)
    d = next(d for d in range(0, 10) if all(abs(round(t, d) - t) <= 1e-9 * scale for t in ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{d}f'))
    return [f'%.{d}f' % t for t in ticks]


def save(fig, out, stem):
    if stem == STEM:
        fig.canvas.draw()
        changed = {}
        for ax in fig.axes:
            if ax.yaxis.get_offset_text().get_text():
                changed[ax.get_title(loc='center')] = plain_decimals(ax)
        print('plain decimal labels:', changed, flush=True)
        fig.canvas.draw(); fig.set_layout_engine('none'); r = fig.canvas.get_renderer()
        assert not any(ax.yaxis.get_offset_text().get_text() for ax in fig.axes)
        ax = [a for a in fig.axes if a.get_xlabel() == 'Updates (thousands)'][0]; xl = ax.xaxis.label
        bb = xl.get_window_extent(r).transformed(fig.transFigure.inverted())
        row = [a for a in fig.axes if abs(a.get_position().y0 - ax.get_position().y0) < 1e-6]
        x0, x1 = min(a.get_position().x0 for a in row), max(a.get_position().x1 for a in row)
        xl.set_color('none')
        fig.text((x0 + x1) / 2, (bb.y0 + bb.y1) / 2, 'Updates (thousands)', ha='center', va='center', fontsize=xl.get_fontsize())
    return _save(fig, out, stem)


S2.save = save
sys.argv = [sys.argv[0], '--out', OUT, STEM]
S2.main()
