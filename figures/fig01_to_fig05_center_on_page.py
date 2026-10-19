#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np
import pymupdf

DPI = 300; Z = DPI / 72; PAD = 1.5


def ink(page, clip=None):
    pix = page.get_pixmap(dpi=DPI, clip=clip, alpha=False)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return (a < 245).any(axis=2)


def visible_spans(page):
    out = []
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                if s['text'].strip() and s.get('alpha', 255) != 0:
                    out.append((pymupdf.Rect(s['bbox']), s['text']))
    return out


def centre(src, out_dir):
    doc = pymupdf.open(src); pg = doc[0]; W, H = pg.rect.width, pg.rect.height
    axes = [p['rect'] for p in pg.get_drawings()
            if p.get('fill') is not None and tuple(round(c, 2) for c in p['fill']) == (1.0, 1.0, 1.0)
            and 30 < p['rect'].width < W - 5 and p['rect'].height > 30]
    gl, gr = min(r.x0 for r in axes), max(r.x1 for r in axes); gc = (gl + gr) / 2
    ax_bottom = max(r.y1 for r in axes)
    spans = visible_spans(pg)
    xl = [r for r, t in spans if 'Updates' in t]
    assert len(xl) == 1, (src, xl)
    xl = xl[0]; xl_c = (xl.x0 + xl.x1) / 2
    xl_on_grid = abs(xl_c - gc) < 2.0
    m = ink(pg); rows = m.any(axis=1)
    def blank_gap(y_from, y_to):
        r0, r1 = int(y_from * Z), int(y_to * Z); best = (0, None); run = None
        for i in range(r0, r1):
            if not rows[i]:
                run = i if run is None else run
                if i - run + 1 > best[0]: best = (i - run + 1, (run + i) / 2)
            else: run = None
        assert best[1] is not None, (src, y_from, y_to)
        return best[1] / Z, best[0] / Z
    leg = [r for r, t in spans if r.y0 > xl.y0 + 0.5 * xl.height]
    leg_top = min(r.y0 + 0.5 * r.height for r in leg)
    if xl_on_grid:
        cut, gap = blank_gap(xl.y0 + 0.5 * xl.height, leg_top)
    else:
        cut, gap = blank_gap(ax_bottom, xl.y0 + 0.6 * xl.height)
    top = m[: int(cut * Z)]; bot = m[int(cut * Z):]
    tc = np.where(top.any(axis=0))[0]; bc = np.where(bot.any(axis=0))[0]
    cl, cr = tc[0] / Z, (tc[-1] + 1) / Z; bl, br = bc[0] / Z, (bc[-1] + 1) / Z
    half = max(gc - cl, cr - gc, (br - bl) / 2) + PAD
    Wn = 2 * half
    dx_top = Wn / 2 - gc; dx_bot = Wn / 2 - (bl + br) / 2
    out = pymupdf.open(); page = out.new_page(width=Wn, height=H)
    page.show_pdf_page(pymupdf.Rect(dx_top, 0, dx_top + W, cut), doc, 0, clip=pymupdf.Rect(0, 0, W, cut))
    page.show_pdf_page(pymupdf.Rect(dx_bot, cut, dx_bot + W, H), doc, 0, clip=pymupdf.Rect(0, cut, W, H))
    dst = Path(out_dir) / Path(src).name
    out.set_metadata({'title': Path(src).stem, 'author': '', 'creator': 'center_figures.py', 'producer': 'PyMuPDF'})
    out.save(dst, garbage=4, deflate=True, no_new_id=True)
    chk = pymupdf.open(dst)[0]; m2 = ink(chk)
    lost = int(m.sum()) - int(m2.sum())
    return dict(src=str(src), out=str(dst), W=round(W, 2), H=round(H, 2), Wn=round(Wn, 2), grid=[round(gl, 1), round(gr, 1)],
                grid_offset_before=round(gc - W / 2, 1), xlabel_on_grid=xl_on_grid, cut=round(cut, 1), gap=round(gap, 2),
                top_ink=[round(cl, 1), round(cr, 1)], bottom_ink=[round(bl, 1), round(br, 1)], ink_px_old=int(m.sum()),
                ink_px_new=int(m2.sum()), ink_px_lost=lost)


if __name__ == '__main__':
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    res = [centre(s, od) for s in sys.argv[2:]]
    print(json.dumps(res, indent=1))
