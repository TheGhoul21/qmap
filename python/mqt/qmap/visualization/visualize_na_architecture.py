# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Visualization of zoned neutral atom architectures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.figure


def _slm_site_coords(slm: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Return (xs, ys) lists of all trap site coordinates in an SLM."""
    loc_x, loc_y = slm["location"]
    sep_x, sep_y = slm["site_separation"]
    xs = [loc_x + c * sep_x for c in range(slm["c"]) for _ in range(slm["r"])]
    ys = [loc_y + r * sep_y for _ in range(slm["c"]) for r in range(slm["r"])]
    return xs, ys


def _slm_bbox(slm: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) bounding box of an SLM array."""
    loc_x, loc_y = slm["location"]
    sep_x, sep_y = slm["site_separation"]
    return (
        float(loc_x),
        float(loc_y),
        float(loc_x + (slm["c"] - 1) * sep_x),
        float(loc_y + (slm["r"] - 1) * sep_y),
    )


def _zone_bbox(slms: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """Return bounding box for a group of SLMs."""
    boxes = [_slm_bbox(s) for s in slms]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def visualize_architecture(
    arch: str | dict[str, Any],
    figsize: tuple[float, float] | None = None,
    show_slm_ids: bool = True,
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    """Visualize the physical layout of a zoned neutral atom architecture.

    Draws storage zones, entanglement zones, Rydberg laser ranges, and SLM
    trap sites on a 2D coordinate plot. AODs are dynamic and therefore only
    listed in the legend with their grid dimensions.

    Args:
        arch: Architecture as a JSON string or dict — the same format accepted
              by ``ZonedNeutralAtomArchitecture.from_json_string()``.
        figsize: Figure size (width, height) in inches. Auto-sized if ``None``.
        show_slm_ids: Annotate each SLM array with its ID.
        title: Figure title. Defaults to the architecture name.

    Returns:
        A :class:`matplotlib.figure.Figure` showing the layout.

    Raises:
        ImportError: If ``matplotlib`` is not installed.

    Example::

        from mqt.qmap.visualization import visualize_architecture
        fig = visualize_architecture(arch_json_string)
        fig.savefig("arch.pdf")
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as e:
        msg = "matplotlib is required for architecture visualization. Install it with: pip install matplotlib"
        raise ImportError(msg) from e

    data: dict[str, Any] = json.loads(arch) if isinstance(arch, str) else dict(arch)

    # ── colour palette ────────────────────────────────────────────────────────
    STORAGE_BG = "#cde8ff"
    STORAGE_EDGE = "#2980b9"
    STORAGE_DOT = "#1a5276"

    # Each entanglement zone has two SLMs; we give them distinct dot colours so
    # pairs are easy to distinguish (atoms from different SLMs interact via CZ).
    ENT_BG = "#fde8c8"
    ENT_EDGE = "#e67e22"
    ENT_DOTS = ["#c0392b", "#922b21"]

    RYDBERG_BG = "#ede0f5"
    RYDBERG_EDGE = "#7d3c98"

    PAD = 4  # padding around zone background rectangles (in architecture units)

    # ── figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize or (14, 9))
    ax.set_aspect("equal")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(title or data.get("name", "Architecture"))
    ax.grid(visible=True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    # y increases with row index in the architecture (screen-style coordinates).
    # Flip the y-axis so row 0 is at the top, matching the physical convention
    # used in NAViz and the architecture JSON.
    ax.invert_yaxis()

    # ── Rydberg ranges (drawn first so they sit behind everything) ───────────
    rydberg_legend = False
    for i, rr in enumerate(data.get("rydberg_range", [])):
        rx0, ry0 = float(rr[0][0]), float(rr[0][1])
        rx1, ry1 = float(rr[1][0]), float(rr[1][1])
        ax.add_patch(
            Rectangle(
                (rx0, ry0),
                rx1 - rx0,
                ry1 - ry0,
                linewidth=1.5,
                edgecolor=RYDBERG_EDGE,
                facecolor=RYDBERG_BG,
                linestyle="--",
                alpha=0.5,
                zorder=1,
            )
        )
        # Place the label at the bottom-right corner to avoid overlapping zone labels.
        ax.text(
            rx1 - 2,
            ry1 - 2,
            f"Rydberg range {i}  [{rx0},{ry0}]→[{rx1},{ry1}]",
            fontsize=7,
            color=RYDBERG_EDGE,
            ha="right",
            va="bottom",
            zorder=6,
        )
        rydberg_legend = True

    # ── storage zones ────────────────────────────────────────────────────────
    for zone_idx, zone in enumerate(data.get("storage_zones", [])):
        slms = zone["slms"]
        zx0, zy0, zx1, zy1 = _zone_bbox(slms)
        ax.add_patch(
            Rectangle(
                (zx0 - PAD, zy0 - PAD),
                (zx1 - zx0) + 2 * PAD,
                (zy1 - zy0) + 2 * PAD,
                linewidth=1.5,
                edgecolor=STORAGE_EDGE,
                facecolor=STORAGE_BG,
                alpha=0.7,
                zorder=2,
            )
        )
        ax.text(
            zx0 - PAD + 1,
            zy0 - PAD + 1,
            f"Storage zone {zone_idx}",
            fontsize=8,
            color=STORAGE_EDGE,
            va="top",
            fontweight="bold",
            zorder=7,
        )
        for slm in slms:
            xs, ys = _slm_site_coords(slm)
            ax.scatter(xs, ys, s=3, c=STORAGE_DOT, linewidths=0, zorder=4)
            if show_slm_ids:
                bx0, by0, bx1, by1 = _slm_bbox(slm)
                ax.text(
                    (bx0 + bx1) / 2,
                    (by0 + by1) / 2,
                    f"SLM {slm['id']}\n{slm['r']}r×{slm['c']}c",
                    fontsize=6,
                    ha="center",
                    va="center",
                    color=STORAGE_EDGE,
                    alpha=0.7,
                    zorder=5,
                )

    # ── entanglement zones ───────────────────────────────────────────────────
    for zone_idx, zone in enumerate(data.get("entanglement_zones", [])):
        slms = zone["slms"]  # always exactly 2
        zx0, zy0, zx1, zy1 = _zone_bbox(slms)
        ax.add_patch(
            Rectangle(
                (zx0 - PAD, zy0 - PAD),
                (zx1 - zx0) + 2 * PAD,
                (zy1 - zy0) + 2 * PAD,
                linewidth=1.5,
                edgecolor=ENT_EDGE,
                facecolor=ENT_BG,
                alpha=0.7,
                zorder=2,
            )
        )
        ax.text(
            zx0 - PAD + 1,
            zy0 - PAD + 1,
            f"Entanglement zone {zone_idx}",
            fontsize=8,
            color=ENT_EDGE,
            va="top",
            fontweight="bold",
            zorder=7,
        )
        for slm_idx, (slm, dot_color) in enumerate(zip(slms, ENT_DOTS)):
            xs, ys = _slm_site_coords(slm)
            ax.scatter(xs, ys, s=10, c=dot_color, linewidths=0, zorder=4)
            if show_slm_ids:
                bx0, by0, bx1, by1 = _slm_bbox(slm)
                # When the two SLMs nearly overlap (typical case), offset labels
                # to the upper/lower third of the zone so they don't stack.
                label_y_frac = 0.33 if slm_idx == 0 else 0.67
                ax.text(
                    (bx0 + bx1) / 2,
                    by0 + (by1 - by0) * label_y_frac,
                    f"SLM {slm['id']}  {slm['r']}r×{slm['c']}c",
                    fontsize=6,
                    ha="center",
                    va="center",
                    color=dot_color,
                    alpha=0.85,
                    zorder=5,
                )

    # ── legend ───────────────────────────────────────────────────────────────
    legend_handles: list[mpatches.Patch] = [
        mpatches.Patch(facecolor=STORAGE_BG, edgecolor=STORAGE_EDGE, label="Storage zone (SLM sites)"),
        mpatches.Patch(facecolor=ENT_BG, edgecolor=ENT_EDGE, label="Entanglement zone (SLM pair)"),
    ]
    if rydberg_legend:
        legend_handles.append(
            mpatches.Patch(
                facecolor=RYDBERG_BG,
                edgecolor=RYDBERG_EDGE,
                linestyle="--",
                label="Rydberg laser range",
            )
        )
    for aod in data.get("aods", []):
        legend_handles.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="none",
                label=f"AOD {aod['id']}: {aod['r']}×{aod['c']} grid, site sep = {aod['site_separation']} μm",
            )
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.autoscale_view()
    fig.tight_layout()
    return fig
