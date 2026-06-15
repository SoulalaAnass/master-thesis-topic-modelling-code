"""
Publication-quality dependency parse tree SVGs.
Style: words on baseline · POS tags below · rectangular arcs above · dark navy.
Matches the clean scientific figure style (no arc labels, no colour fills).
"""
import os, re

BASE = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(ROOT, "figures")
# dependency-parse HTML produced by step4_dependency_parses.py
HTML_DIR = os.path.join(ROOT, "figures", "parse_html")
os.makedirs(OUT, exist_ok=True)

# ── visual constants ─────────────────────────────────────────────────────────
INK      = "#2d3561"           # dark navy
BG       = "#ffffff"
FONT     = "Arial, Helvetica, sans-serif"
W_SZ     = 15                  # word font size (px)
P_SZ     = 12                  # POS tag font size (px)
CHAR_W   = 8.8                 # avg px per character at W_SZ
MIN_TOK  = 45                  # min token slot width (px)
GAP      = 32                  # horizontal gap between tokens (px)
ARC_MIN  = 32                  # arc height for span-1 arcs (px)
ARC_STP  = 26                  # additional px per span unit
ARC_BASE = 12                  # px gap from word top to arc attachment
PAD_X    = 40                  # left/right canvas padding
PAD_TOP  = 18                  # breathing room above highest arc
PAD_BOT  = 18                  # breathing room below POS tags
AH_W     = 7                   # arrowhead half-width
AH_H     = 8                   # arrowhead height

# ── parse displaCy HTML ──────────────────────────────────────────────────────

def read_svg(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    m = re.search(r'(<svg\s[^>]*displacy[^>]*>[\s\S]*?</svg>)', raw)
    return m.group(1) if m else ""

def parse(svg):
    """Extract ordered token list and dependency edges from displaCy SVG."""
    tokens = {}
    for blk in re.finditer(r'<text class="displacy-token"[^>]*>([\s\S]*?)</text>', svg):
        wm = re.search(r'<tspan class="displacy-word"[^>]* x="([\d.]+)"[^>]*>(.*?)</tspan>', blk.group(1))
        tm = re.search(r'<tspan class="displacy-tag"[^>]* x="[\d.]+"[^>]*>(.*?)</tspan>',    blk.group(1))
        if wm and tm:
            tokens[round(float(wm.group(1)))] = {"word": wm.group(2), "pos": tm.group(1)}

    xs   = sorted(tokens)
    snap = lambda v: min(xs, key=lambda x: abs(x - v))
    edges = []
    for blk in re.finditer(r'<g class="displacy-arrow">([\s\S]*?)</g>', svg):
        pm = re.search(
            r'd="M\s*([\d.]+),[\d.]+\s+[\d.]+,[\d.]+\s+([\d.]+),[\d.]+\s+[\d.]+,[\d.]+"',
            blk.group(1))
        am = re.search(r'class="displacy-arrowhead" d="M\s*([\d.]+)', blk.group(1))
        lm = re.search(r'class="displacy-label"[^>]*>(.*?)</textPath>', blk.group(1))
        if pm and am:
            xa, xb = float(pm.group(1)), float(pm.group(2))
            ah = float(am.group(1))
            if abs(ah - xa) < abs(ah - xb):
                dep_x, head_x = snap(xa), snap(xb)
            else:
                dep_x, head_x = snap(xb), snap(xa)
            edges.append((head_x, dep_x, lm.group(1) if lm else ""))
    return tokens, edges

# ── layout ───────────────────────────────────────────────────────────────────

def token_centers(tokens):
    """Compute centre-x for every token, proportional to word length."""
    xs  = sorted(tokens)
    cxs = {}
    cur = PAD_X
    for tx in xs:
        slot = max(len(tokens[tx]["word"]) * CHAR_W, MIN_TOK)
        cxs[tx] = cur + slot / 2
        cur += slot + GAP
    return cxs, cur - GAP + PAD_X      # (centres dict, total canvas width)

def arc_h(span):
    """Arc height in px for a dependency spanning <span> tokens apart."""
    return ARC_MIN + (span - 1) * ARC_STP

# ── render ───────────────────────────────────────────────────────────────────

def render(tokens, edges, filename):
    xs        = sorted(tokens)
    rank      = {x: i for i, x in enumerate(xs)}
    cxs, W    = token_centers(tokens)

    arcs      = [(hx, dx, lbl, arc_h(abs(rank[hx] - rank[dx])))
                 for hx, dx, lbl in edges]
    max_arc_h = max((a[3] for a in arcs), default=ARC_MIN)

    # ── y positions ──
    word_y    = PAD_TOP + max_arc_h + ARC_BASE + W_SZ     # word text baseline
    pos_y     = word_y + P_SZ + 7                          # POS tag baseline
    H         = pos_y + PAD_BOT
    attach_y  = word_y - W_SZ - ARC_BASE                  # arc attachment line

    out = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     viewBox="0 0 {W:.0f} {H:.0f}"',
        f'     width="{W:.0f}" height="{H:.0f}">',
        f'  <rect width="100%" height="100%" fill="{BG}"/>',
        f'  <g font-family="{FONT}" fill="{INK}" text-anchor="middle">',
    ]

    # ── arcs (draw shorter ones first so taller ones render on top) ──
    for hx, dx, _, h in sorted(arcs, key=lambda a: a[3]):
        x1   = cxs[hx]
        x2   = cxs[dx]
        top  = attach_y - h          # top of the rectangular arc

        # Rectangular L-shaped path: head up → horizontal → dep down
        out.append(
            f'    <path d="M {x1:.1f},{attach_y:.1f}'
            f' L {x1:.1f},{top:.1f}'
            f' L {x2:.1f},{top:.1f}'
            f' L {x2:.1f},{attach_y - AH_H:.1f}"'
            f' stroke="{INK}" stroke-width="1.6"'
            f' stroke-linejoin="miter" fill="none"/>'
        )
        # Filled arrowhead pointing DOWN at dep
        out.append(
            f'    <polygon'
            f' points="{x2 - AH_W:.1f},{attach_y - AH_H:.1f}'
            f' {x2 + AH_W:.1f},{attach_y - AH_H:.1f}'
            f' {x2:.1f},{attach_y:.1f}"'
            f' fill="{INK}"/>'
        )

    # ── words & POS tags ──
    for tx in xs:
        cx   = cxs[tx]
        word = tokens[tx]["word"]
        pos  = tokens[tx]["pos"]
        out.append(
            f'    <text x="{cx:.1f}" y="{word_y:.1f}"'
            f' font-size="{W_SZ}" font-weight="600">{word}</text>'
        )
        out.append(
            f'    <text x="{cx:.1f}" y="{pos_y:.1f}"'
            f' font-size="{P_SZ}">{pos}</text>'
        )

    out += ["  </g>", "</svg>"]

    path = os.path.join(OUT, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Written: {filename}  ({W:.0f} × {H:.0f})")

# ── generate all 6 ───────────────────────────────────────────────────────────

configs = [
    ("topic_09_recommendation_formula.svg", "topic_09_sentence_1_analysis.html"),
    ("topic_07_nicht_mal_minimiser.svg",    "topic_07_sentence_1_analysis.html"),
    ("topic_05_wir_fuehlen_uns.svg",        "topic_05_sentence_2_analysis.html"),
    ("topic_03_aufnahme_process.svg",       "topic_03_sentence_3_analysis.html"),
    ("topic_04_enorm_bedauerlich.svg",      "topic_04_sentence_1_analysis.html"),  # 7 tokens
    ("topic_00_complaint_structure.svg",    "topic_00_sentence_1_analysis.html"),  # 18 tokens
]

for fname, html in configs:
    svg_str = read_svg(os.path.join(HTML_DIR, html))
    if not svg_str:
        print(f"  [skip] {html}")
        continue
    toks, edges = parse(svg_str)
    render(toks, edges, fname)

print("\nAll done!")
