"""AAAI single-column figure style: 3.3 in wide, fonts >= 8 pt, TrueType embedded, colorblind-safe colours."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402

WIDTH = 3.3
SETTING = {"quiet": "#2457A6", "balanced": "#7A4AA8", "supportive": "#1F7A58"}
# Quiet and Balanced are close under protan CVD, so settings always also get a distinct marker and a text label.
SETTING_MARKER = {"quiet": "o", "balanced": "s", "supportive": "^"}
# Okabe-Ito for everything that is not a setting.
OI = {"orange": "#E69F00", "vermillion": "#D55E00", "sky": "#56B4E9", "black": "#000000", "gray": "#6E6E6E", "yellow": "#F0E442"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "grid.linewidth": 0.4, "grid.color": "#DDDDDD",
    "lines.linewidth": 1.4, "savefig.dpi": 300,
})


def save(fig, name):
    C.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG / f"{name}.pdf")
    fig.savefig(C.FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    return [str((C.FIG / f"{name}.{e}").relative_to(C.REPO)) for e in ("pdf", "png")]
