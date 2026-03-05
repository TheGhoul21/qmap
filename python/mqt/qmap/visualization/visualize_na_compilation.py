# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Visualization and animation of zoned neutral atom compilation results."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.animation
    import matplotlib.figure
    import mqt.core.ir


# ── resource guard ─────────────────────────────────────────────────────────────


def compilation_guard(
    qc: "mqt.core.ir.QuantumComputation",
    compiler: str = "agnostic",
    max_nodes: int = 10_000_000,
    max_animation_frames: int = 200,
    raise_on_danger: bool = False,
) -> dict[str, Any]:
    """Estimate compilation cost and warn about potential resource issues.

    Call this *before* compiling to catch configurations that could exhaust
    RAM or take a very long time.

    Args:
        qc: The quantum circuit to compile.
        compiler: ``"agnostic"`` (RoutingAgnosticCompiler) or
                  ``"aware"`` (RoutingAwareCompiler).
        max_nodes: The ``max_nodes`` setting that will be passed to
                   RoutingAwareCompiler (ignored for agnostic).
        max_animation_frames: Warn if the estimated animation frame count
                              exceeds this value.
        raise_on_danger: If ``True``, raise ``ResourceWarning`` instead of
                         just printing warnings when dangerous settings are
                         detected.

    Returns:
        A dict with keys ``n_qubits``, ``n_2q_gates``, ``estimated_layers``,
        ``estimated_frames``, ``warnings``, ``safe``.
    """
    n_qubits: int = qc.num_qubits
    n_2q: int = sum(1 for op in qc if op.num_controls + op.num_targets == 2)

    # Rough layer estimate: ASAP scheduling parallelises at most n_qubits//2
    # gates per layer, so n_layers ≈ n_2q / (n_qubits // 2).
    max_parallel = max(1, n_qubits // 2)
    estimated_layers = max(1, (n_2q + max_parallel - 1) // max_parallel)
    estimated_frames = 2 * estimated_layers + 1

    issues: list[str] = []

    if compiler == "aware":
        mem_gb = max_nodes * 120 / 1e9
        if mem_gb > 4.0:
            issues.append(
                f"RoutingAwareCompiler: max_nodes={max_nodes:,} → ~{mem_gb:.1f} GB peak RAM. "
                f"Reduce max_nodes (e.g. 500_000 → ~60 MB) or switch to the agnostic compiler."
            )
        elif mem_gb > 1.0:
            issues.append(
                f"RoutingAwareCompiler: max_nodes={max_nodes:,} → ~{mem_gb:.1f} GB peak RAM."
            )
        if n_qubits > 24:
            issues.append(
                f"RoutingAwareCompiler with {n_qubits} qubits may be slow or exhaust max_nodes. "
                f"Consider RoutingAgnosticCompiler for large circuits."
            )

    if estimated_frames > max_animation_frames:
        issues.append(
            f"Estimated {estimated_frames} animation frames (>{max_animation_frames}). "
            f"animate_compilation() may be slow / produce a large file. "
            f"Use visualize_compilation_step() to inspect individual frames instead."
        )

    safe = len(issues) == 0
    result: dict[str, Any] = {
        "n_qubits": n_qubits,
        "n_2q_gates": n_2q,
        "estimated_layers": estimated_layers,
        "estimated_frames": estimated_frames,
        "warnings": issues,
        "safe": safe,
    }

    for msg in issues:
        if raise_on_danger:
            raise ResourceWarning(msg)
        warnings.warn(msg, ResourceWarning, stacklevel=2)

    return result


# ── shared geometry helpers ────────────────────────────────────────────────────


def _slm_coords(slm: dict[str, Any]) -> dict[str, Any]:
    """Pre-compute geometry for an SLM: site positions and bounding box."""
    loc_x, loc_y = slm["location"]
    sep_x, sep_y = slm["site_separation"]
    nrows, ncols = slm["r"], slm["c"]
    sites = [
        [(loc_x + c * sep_x, loc_y + r * sep_y) for c in range(ncols)]
        for r in range(nrows)
    ]
    return {
        "id": slm["id"],
        "loc": (loc_x, loc_y),
        "sep": (sep_x, sep_y),
        "nrows": nrows,
        "ncols": ncols,
        "sites": sites,
        "bbox": (loc_x, loc_y, loc_x + (ncols - 1) * sep_x, loc_y + (nrows - 1) * sep_y),
    }


def _build_slm_map(arch: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return a map from slm_id → geometry for all SLMs in the architecture."""
    slm_map: dict[int, dict[str, Any]] = {}
    for zone in arch.get("storage_zones", []):
        for slm in zone["slms"]:
            slm_map[slm["id"]] = _slm_coords(slm)
    for zone in arch.get("entanglement_zones", []):
        for slm in zone["slms"]:
            slm_map[slm["id"]] = _slm_coords(slm)
    return slm_map


def _qubit_xy(slm_map: dict[int, dict[str, Any]], site: list[int]) -> tuple[float, float]:
    """Convert a [slm_id, row, col] site to (x, y) global coordinates."""
    slm_id, row, col = site
    return slm_map[slm_id]["sites"][row][col]


# ── colour helpers ─────────────────────────────────────────────────────────────

_QUBIT_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#a9a9a9", "#e6beff", "#fffac8", "#fabebe",
]


def _qubit_color(q: int) -> str:
    return _QUBIT_PALETTE[q % len(_QUBIT_PALETTE)]


# ── background layer (zones + Rydberg ranges) ──────────────────────────────────

def _draw_background(ax: Any, arch: dict[str, Any], laser_active: bool = False) -> None:
    """Draw zones and Rydberg ranges on *ax* (no atoms).

    When *laser_active* is True the Rydberg range is rendered as a bright
    glowing overlay to indicate that the laser beam is firing.
    """
    from matplotlib.patches import Rectangle
    PAD = 4

    for rr in arch.get("rydberg_range", []):
        rx0, ry0 = float(rr[0][0]), float(rr[0][1])
        rx1, ry1 = float(rr[1][0]), float(rr[1][1])
        if laser_active:
            # Bright glow: solid fill + thick border to mimic laser illumination
            ax.add_patch(Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                linewidth=2.5, edgecolor="#e74c3c", facecolor="#ff6b6b",
                linestyle="-", alpha=0.30, zorder=1,
            ))
            # Second pass: inner bloom
            ax.add_patch(Rectangle(
                (rx0 + 1, ry0 + 1), rx1 - rx0 - 2, ry1 - ry0 - 2,
                linewidth=0, facecolor="#ff0000",
                alpha=0.10, zorder=1,
            ))
            ax.text(
                (rx0 + rx1) / 2, ry0 + (ry1 - ry0) * 0.15,
                "Rydberg laser \u26a1",
                fontsize=7, ha="center", va="center",
                color="#c0392b", fontweight="bold", zorder=8,
            )
        else:
            ax.add_patch(Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                linewidth=1.2, edgecolor="#7d3c98", facecolor="#ede0f5",
                linestyle="--", alpha=0.4, zorder=1,
            ))

    def _zone_bbox(slms: list[dict[str, Any]]) -> tuple[float, float, float, float]:
        boxes = [_slm_coords(s)["bbox"] for s in slms]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    for zi, zone in enumerate(arch.get("storage_zones", [])):
        zx0, zy0, zx1, zy1 = _zone_bbox(zone["slms"])
        ax.add_patch(Rectangle(
            (zx0 - PAD, zy0 - PAD), (zx1 - zx0) + 2 * PAD, (zy1 - zy0) + 2 * PAD,
            linewidth=1.2, edgecolor="#2980b9", facecolor="#cde8ff", alpha=0.55, zorder=2,
        ))
        ax.text(zx0 - PAD + 1, zy0 - PAD + 1, f"Storage {zi}",
                fontsize=7, color="#2980b9", va="top", fontweight="bold", zorder=7)

    for zi, zone in enumerate(arch.get("entanglement_zones", [])):
        zx0, zy0, zx1, zy1 = _zone_bbox(zone["slms"])
        if laser_active:
            # Entanglement zone glows red when laser fires
            ax.add_patch(Rectangle(
                (zx0 - PAD, zy0 - PAD), (zx1 - zx0) + 2 * PAD, (zy1 - zy0) + 2 * PAD,
                linewidth=2, edgecolor="#e74c3c", facecolor="#fadbd8",
                alpha=0.85, zorder=2,
            ))
            ax.text(zx0 - PAD + 1, zy0 - PAD + 1, f"Entanglement {zi}",
                    fontsize=7, color="#c0392b", va="top", fontweight="bold", zorder=7)
        else:
            ax.add_patch(Rectangle(
                (zx0 - PAD, zy0 - PAD), (zx1 - zx0) + 2 * PAD, (zy1 - zy0) + 2 * PAD,
                linewidth=1.2, edgecolor="#e67e22", facecolor="#fde8c8", alpha=0.55, zorder=2,
            ))
            ax.text(zx0 - PAD + 1, zy0 - PAD + 1, f"Entanglement {zi}",
                    fontsize=7, color="#e67e22", va="top", fontweight="bold", zorder=7)

    slm_map = _build_slm_map(arch)
    for slm in slm_map.values():
        xs = [slm["sites"][r][c][0] for r in range(slm["nrows"]) for c in range(slm["ncols"])]
        ys = [slm["sites"][r][c][1] for r in range(slm["nrows"]) for c in range(slm["ncols"])]
        ax.scatter(xs, ys, s=2, c="#bbbbbb", linewidths=0, zorder=3, alpha=0.5)


# ── single frame render ────────────────────────────────────────────────────────

def _render_frame(
    ax: Any,
    arch: dict[str, Any],
    slm_map: dict[int, dict[str, Any]],
    debug: dict[str, Any],
    frame_idx: int,
) -> None:
    """
    Frame layout — placement has 2*n_layers+1 entries:
      frame 2*i   → placement[2*i]   atoms in storage/transit; routing[2*i] = loading arrows
      frame 2*i+1 → placement[2*i+1] atoms in ent. zone; CZ layer i fires
    Final frame (2*n_layers) → placement[2*n_layers] all back in storage.
    """
    n_layers: int = debug["n_layers"]
    placements: list[list[list[int]]] = debug["placement"]
    routing: list[list[list[int]]] = debug["routing"]
    tq_layers: list[list[list[int]]] = debug["two_qubit_layers"]
    reuse: list[list[int]] = debug["reuse_qubits"]
    n_qubits: int = debug["n_qubits"]

    total_frames = 2 * n_layers + 1
    frame_idx = max(0, min(frame_idx, total_frames - 1))

    layer_idx = frame_idx // 2          # which CZ layer this frame belongs to
    is_entanglement_frame = (frame_idx % 2 == 1)
    current_placement = placements[frame_idx]  # frame index == placement index

    _draw_background(ax, arch, laser_active=is_entanglement_frame and layer_idx < n_layers)

    # ── qubit atoms ──────────────────────────────────────────────────────
    # Detect crowded atoms: if multiple qubits share the exact same site,
    # stagger their labels so they don't all print on top of each other.
    from collections import defaultdict
    site_counts: dict[tuple[float, float], int] = defaultdict(int)
    site_offsets: dict[tuple[float, float], int] = defaultdict(int)
    for q in range(n_qubits):
        xy = _qubit_xy(slm_map, current_placement[q])
        site_counts[xy] += 1

    for q in range(n_qubits):
        x, y = _qubit_xy(slm_map, current_placement[q])
        ax.scatter([x], [y], s=80, c=[_qubit_color(q)], zorder=8, linewidths=0.8, edgecolors="white")
        # For crowded sites, place label offset from the dot; otherwise center it.
        xy = (x, y)
        idx = site_offsets[xy]
        site_offsets[xy] += 1
        if site_counts[xy] > 1:
            # Spread labels in a ring around the dot (offset by ~4 data units)
            import math
            angle = 2 * math.pi * idx / site_counts[xy]
            dx = 5 * math.cos(angle)
            dy = 5 * math.sin(angle)
            ax.annotate(
                str(q),
                xy=(x, y), xytext=(x + dx, y + dy),
                fontsize=5, ha="center", va="center",
                color=_qubit_color(q), fontweight="bold", zorder=9,
                arrowprops=dict(arrowstyle="-", color=_qubit_color(q), lw=0.5),
            )
        else:
            ax.text(x, y, str(q), fontsize=5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=9)

    # ── active CZ pairs (entanglement frames) ────────────────────────────
    # The zone glow already signals the laser is on; draw a thin bond line
    # between each entangled pair so individual gate partners are visible.
    if is_entanglement_frame and layer_idx < n_layers:
        for pair in tq_layers[layer_idx]:
            q0, q1 = pair
            x0, y0 = _qubit_xy(slm_map, current_placement[q0])
            x1, y1 = _qubit_xy(slm_map, current_placement[q1])
            ax.plot([x0, x1], [y0, y1], "-", color="#c0392b", linewidth=2.5, alpha=0.9, zorder=6)

    # ── movement arrows (storage frames) ─────────────────────────────────
    # routing[frame_idx] = loading move: placement[frame_idx] → placement[frame_idx+1]
    if not is_entanglement_frame and frame_idx < len(routing):
        src = placements[frame_idx]
        dst = placements[frame_idx + 1]
        # Assign a small arc bend per qubit index so overlapping arrows fan out.
        moving_qubits = [q for group in routing[frame_idx] for q in group
                         if _qubit_xy(slm_map, src[q]) != _qubit_xy(slm_map, dst[q])]
        n_moving = len(moving_qubits)
        for rank, q in enumerate(moving_qubits):
            x0, y0 = _qubit_xy(slm_map, src[q])
            x1, y1 = _qubit_xy(slm_map, dst[q])
            # Spread arc curvature from -0.3 to +0.3 across all moving atoms.
            rad = 0.0 if n_moving == 1 else -0.3 + 0.6 * rank / (n_moving - 1)
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=_qubit_color(q),
                    lw=1.2, mutation_scale=10,
                    connectionstyle=f"arc3,rad={rad:.2f}",
                ),
                zorder=5,
            )
            # Small qubit label at midpoint of arrow
            import math
            mx = (x0 + x1) / 2 + 3 * math.sin(rad * math.pi)
            my = (y0 + y1) / 2 - 3 * math.cos(rad * math.pi)
            ax.text(mx, my, str(q), fontsize=4, ha="center", va="center",
                    color=_qubit_color(q), fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    # ── reuse markers ─────────────────────────────────────────────────────
    if is_entanglement_frame and layer_idx < len(reuse):
        for q in reuse[layer_idx]:
            x, y = _qubit_xy(slm_map, current_placement[q])
            ax.scatter([x], [y], s=200, facecolors="none",
                       edgecolors="#2ecc71", linewidths=1.5, zorder=10)

    # ── single-qubit gate indicators (storage frames) ─────────────────────
    # single_qubit_layers[i] contains the 1Q gates executed while atoms
    # sit in storage *before* two-qubit gate layer i (or as a trailing layer).
    sq_layers: list[list[dict[str, Any]]] = debug.get("single_qubit_layers", [])
    sq_on_qubit: dict[int, list[tuple[str, list[float]]]] = {}
    if not is_entanglement_frame and layer_idx < len(sq_layers):
        # Build per-qubit list of (gate_name, params) for this layer
        for op in sq_layers[layer_idx]:
            entry = (op["name"], op.get("params", []))
            for q in op["qubits"]:
                sq_on_qubit.setdefault(q, []).append(entry)
        for q, ops in sq_on_qubit.items():
            x, y = _qubit_xy(slm_map, current_placement[q])
            # Yellow dashed ring around the atom
            ax.scatter([x], [y], s=260, facecolors="none",
                       edgecolors="#f1c40f", linewidths=1.5,
                       linestyles="dashed", zorder=10)
            # Build label: gate name + formatted angles if present
            parts = []
            for name, params in ops:
                if params:
                    angle_strs = []
                    for p in params:
                        # Express in units of π, rounded to 2 sig figs
                        frac = p / 3.141592653589793
                        if abs(frac - round(frac)) < 1e-6:
                            angle_strs.append(f"{round(frac):.0f}π")
                        elif abs(frac * 2 - round(frac * 2)) < 1e-6:
                            n = round(frac * 2)
                            angle_strs.append(f"{n}/2π")
                        elif abs(frac * 4 - round(frac * 4)) < 1e-6:
                            n = round(frac * 4)
                            angle_strs.append(f"{n}/4π")
                        else:
                            angle_strs.append(f"{p:.2f}")
                    parts.append(f"{name}({','.join(angle_strs)})")
                else:
                    parts.append(name)
            label = "\n".join(parts)
            ax.text(x, y - 4.5, label, fontsize=4, ha="center", va="bottom",
                    color="#b7950b", fontweight="bold", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#f1c40f",
                              alpha=0.85, linewidth=0.8))

    # ── title ─────────────────────────────────────────────────────────────
    if is_entanglement_frame:
        cz_pairs = tq_layers[layer_idx] if layer_idx < n_layers else []
        n_cz = len(cz_pairs)
        ax.set_title(
            f"Layer {layer_idx}  —  {n_cz} CZ gate{'s' if n_cz != 1 else ''}  "
            f"(Rydberg laser \u26a1)\n(frame {frame_idx}/{2 * n_layers})",
            fontsize=9)
    else:
        move_count = sum(len(g) for g in routing[frame_idx]) if frame_idx < len(routing) else 0
        sq_note = f"  |  {len(sq_on_qubit)} qubit{'s' if len(sq_on_qubit) != 1 else ''} with 1Q gates" if sq_on_qubit else ""
        ax.set_title(
            f"Transition {layer_idx}  —  {move_count} atoms moving{sq_note}\n"
            f"(frame {frame_idx}/{2 * n_layers})", fontsize=9)


# ── public API ─────────────────────────────────────────────────────────────────


def visualize_compilation_step(
    debug: dict[str, Any],
    arch: str | dict[str, Any],
    frame: int = 0,
    figsize: tuple[float, float] | None = None,
) -> "matplotlib.figure.Figure":
    """Render a single frame of the compilation animation.

    Each *frame* alternates between two views:

    * **Even frames** (0, 2, 4, …): atoms are in their storage placement
      with movement arrows showing where they will travel next.
    * **Odd frames** (1, 3, 5, …): atoms have arrived at the entanglement
      zone; active CZ pairs are highlighted in red. Reused qubits (those
      that stay in the entanglement zone for the next layer) are circled
      in green.

    The total number of frames is ``2 * n_layers + 1``.

    Args:
        debug: The dictionary returned by ``compiler.debug_info()``.
        arch: Architecture JSON string or dict.
        frame: Frame index to render (0-based).
        figsize: Figure size in inches.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        msg = "matplotlib is required: pip install matplotlib"
        raise ImportError(msg) from e

    arch_data: dict[str, Any] = json.loads(arch) if isinstance(arch, str) else dict(arch)
    slm_map = _build_slm_map(arch_data)

    fig, ax = plt.subplots(figsize=figsize or (14, 9))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.grid(visible=True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    _render_frame(ax, arch_data, slm_map, debug, frame)

    import matplotlib.lines as mlines

    is_ent = (frame % 2 == 1) and (frame // 2 < debug["n_layers"])

    # ── qubit colour swatches ─────────────────────────────────────────────
    qubit_handles = [
        mpatches.Patch(color=_qubit_color(q), label=f"q{q}")
        for q in range(debug["n_qubits"])
    ]

    # ── visual-element legend (always shown) ─────────────────────────────
    element_handles = [
        # zones — always present
        mpatches.Patch(facecolor="#cde8ff", edgecolor="#2980b9",
                       label="storage zone"),
        mpatches.Patch(facecolor="#fde8c8", edgecolor="#e67e22",
                       label="entanglement zone"),
        mpatches.Patch(facecolor="#ede0f5", edgecolor="#7d3c98",
                       linestyle="dashed", label="Rydberg laser range"),
        # atoms
        mlines.Line2D([], [], color="none", marker="o", markersize=8,
                      markerfacecolor="#4a90d9", markeredgecolor="white",
                      label="atom (colour = qubit index)"),
    ]
    if is_ent:
        element_handles += [
            mpatches.Patch(facecolor="#ff6b6b", edgecolor="#e74c3c",
                           alpha=0.7, label="Rydberg laser firing"),
            mlines.Line2D([], [], color="#c0392b", linewidth=2,
                          label="CZ gate bond"),
            mlines.Line2D([], [], color="none", marker="o", markersize=10,
                          markerfacecolor="none", markeredgecolor="#2ecc71",
                          markeredgewidth=1.5, label="reused qubit (stays in ent. zone)"),
        ]
    else:
        element_handles += [
            mlines.Line2D([], [], color="#888888", linewidth=1.5,
                          marker=">", markersize=5,
                          label="atom move (AOD transport)"),
            mlines.Line2D([], [], color="none", marker="o", markersize=10,
                          markerfacecolor="none", markeredgecolor="#f1c40f",
                          markeredgewidth=1.5, linestyle="dashed",
                          label="1Q gate being applied"),
        ]

    # Two-column layout: qubits on the left, elements on the right
    n_q = debug["n_qubits"]
    qubit_legend = ax.legend(
        handles=qubit_handles,
        loc="upper right",
        fontsize=7,
        ncol=max(1, n_q // 8),
        framealpha=0.9,
        title="qubits",
        title_fontsize=7,
    )
    ax.add_artist(qubit_legend)
    ax.legend(
        handles=element_handles,
        loc="upper left",
        fontsize=7,
        framealpha=0.9,
        title="legend",
        title_fontsize=7,
    )

    ax.autoscale_view()
    fig.tight_layout()
    return fig


def animate_compilation(
    debug: dict[str, Any],
    arch: str | dict[str, Any],
    interval: int = 800,
    figsize: tuple[float, float] | None = None,
    repeat: bool = True,
    max_frames: int | None = None,
) -> "matplotlib.animation.FuncAnimation":
    """Create an animated step-through of the full compilation.

    Each frame alternates between storage placements (with movement arrows)
    and entanglement-zone placements (with CZ highlights). Total frames =
    ``2 * n_layers + 1``.

    To save as a video::

        anim = animate_compilation(dbg, arch_json)
        anim.save("out.mp4", fps=2)           # needs ffmpeg
        anim.save("out.gif", writer="pillow", fps=1.5)

    To display in a Jupyter notebook::

        from IPython.display import HTML
        HTML(anim.to_jshtml())

    Args:
        debug: The dictionary returned by ``compiler.debug_info()``.
        arch: Architecture JSON string or dict.
        interval: Milliseconds between frames.
        figsize: Figure size in inches.
        repeat: Whether the animation loops.
        max_frames: Cap the number of frames rendered (useful for large
                    circuits). ``None`` renders all frames.

    Returns:
        A :class:`matplotlib.animation.FuncAnimation`.
    """
    try:
        import matplotlib.animation as animation
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required: pip install matplotlib"
        raise ImportError(msg) from e

    arch_data: dict[str, Any] = json.loads(arch) if isinstance(arch, str) else dict(arch)
    slm_map = _build_slm_map(arch_data)
    n_frames = 2 * debug["n_layers"] + 1
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    fig, ax = plt.subplots(figsize=figsize or (14, 9))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.grid(visible=True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    handles = [mpatches.Patch(color=_qubit_color(q), label=f"q{q}") for q in range(debug["n_qubits"])]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              ncol=max(1, debug["n_qubits"] // 6), framealpha=0.9)

    def _update(frame: int) -> None:
        ax.cla()
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        ax.grid(visible=True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
        _render_frame(ax, arch_data, slm_map, debug, frame)
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  ncol=max(1, debug["n_qubits"] // 6), framealpha=0.9)
        ax.autoscale_view()

    anim = animation.FuncAnimation(fig, _update, frames=n_frames,
                                   interval=interval, repeat=repeat, blit=False)
    fig.tight_layout()
    return anim
