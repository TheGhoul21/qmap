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
import math
import warnings
from collections import defaultdict
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


def _arch_axis_limits(arch: dict[str, Any], pad: float = 6.0) -> tuple[float, float, float, float]:
    """Return fixed (xmin, xmax, ymin, ymax) for the architecture.

    Derived from arch_range if present, otherwise from the union of all zone
    bounding boxes.  A constant pad is added so zone borders are fully visible.
    The same limits are used on every frame so the view never jumps.
    """
    if "arch_range" in arch:
        (rx0, ry0), (rx1, ry1) = arch["arch_range"]
        return float(rx0) - pad, float(rx1) + pad, float(ry0) - pad, float(ry1) + pad

    # Fallback: scan all SLM bboxes
    xs, ys = [], []
    for zone in arch.get("storage_zones", []) + arch.get("entanglement_zones", []):
        for slm in zone["slms"]:
            g = _slm_coords(slm)
            xs += [g["bbox"][0], g["bbox"][2]]
            ys += [g["bbox"][1], g["bbox"][3]]
    if not xs:
        return -10.0, 110.0, -10.0, 70.0
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


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


# ── formatting helpers ─────────────────────────────────────────────────────────

_PI = 3.141592653589793


def _format_angle(p: float) -> str:
    """Format an angle as a human-readable string in multiples of π."""
    frac = p / _PI
    if abs(frac - round(frac)) < 1e-6:
        n = round(frac)
        if n == 0:
            return "0"
        if n == 1:
            return "π"
        if n == -1:
            return "-π"
        return f"{n}π"
    if abs(frac * 2 - round(frac * 2)) < 1e-6:
        n = round(frac * 2)
        return f"{n}/2·π"
    if abs(frac * 4 - round(frac * 4)) < 1e-6:
        n = round(frac * 4)
        return f"{n}/4·π"
    return f"{p:.3f}"


def _format_gate(name: str, params: list[float]) -> str:
    """Format a gate as NAME(angle, …) or just NAME."""
    if params:
        return f"{name}({', '.join(_format_angle(p) for p in params)})"
    return name


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
            ax.add_patch(Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                linewidth=2.5, edgecolor="#e74c3c", facecolor="#ff6b6b",
                linestyle="-", alpha=0.30, zorder=1,
            ))
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
        ax.text(zx1 + PAD - 1, zy0 - PAD + 1, f"Storage {zi}",
                fontsize=7, color="#2980b9", va="top", ha="right", fontweight="bold", zorder=7)

    for zi, zone in enumerate(arch.get("entanglement_zones", [])):
        zx0, zy0, zx1, zy1 = _zone_bbox(zone["slms"])
        if laser_active:
            ax.add_patch(Rectangle(
                (zx0 - PAD, zy0 - PAD), (zx1 - zx0) + 2 * PAD, (zy1 - zy0) + 2 * PAD,
                linewidth=2, edgecolor="#e74c3c", facecolor="#fadbd8",
                alpha=0.85, zorder=2,
            ))
            ax.text(zx1 + PAD - 1, zy0 - PAD + 1, f"Entanglement {zi}",
                    fontsize=7, color="#c0392b", va="top", ha="right", fontweight="bold", zorder=7)
        else:
            ax.add_patch(Rectangle(
                (zx0 - PAD, zy0 - PAD), (zx1 - zx0) + 2 * PAD, (zy1 - zy0) + 2 * PAD,
                linewidth=1.2, edgecolor="#e67e22", facecolor="#fde8c8", alpha=0.55, zorder=2,
            ))
            ax.text(zx1 + PAD - 1, zy0 - PAD + 1, f"Entanglement {zi}",
                    fontsize=7, color="#e67e22", va="top", ha="right", fontweight="bold", zorder=7)

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
) -> dict[str, Any]:
    """Render a single frame onto *ax* and return frame metadata.

    Frame layout — placement has 2*n_layers+1 entries:
      frame 2*i   → placement[2*i]   atoms in storage/transit; routing arrows
      frame 2*i+1 → placement[2*i+1] atoms in ent. zone; CZ layer i fires
    Final frame (2*n_layers) → placement[2*n_layers] all back in storage.

    Returns:
        A dict with keys ``frame_idx``, ``layer_idx``, ``is_entanglement``,
        ``sq_on_qubit`` (for external display helpers).
    """
    n_layers: int = debug["n_layers"]
    placements: list[list[list[int]]] = debug["placement"]
    routing: list[list[list[int]]] = debug["routing"]
    tq_layers: list[list[list[int]]] = debug["two_qubit_layers"]
    reuse: list[list[int]] = debug["reuse_qubits"]
    n_qubits: int = debug["n_qubits"]

    total_frames = 2 * n_layers + 1
    frame_idx = max(0, min(frame_idx, total_frames - 1))

    layer_idx = frame_idx // 2
    is_entanglement_frame = (frame_idx % 2 == 1)
    current_placement = placements[frame_idx]

    _draw_background(ax, arch, laser_active=is_entanglement_frame and layer_idx < n_layers)

    # ── qubit atoms ──────────────────────────────────────────────────────
    site_counts: dict[tuple[float, float], int] = defaultdict(int)
    site_offsets: dict[tuple[float, float], int] = defaultdict(int)
    for q in range(n_qubits):
        xy = _qubit_xy(slm_map, current_placement[q])
        site_counts[xy] += 1

    for q in range(n_qubits):
        x, y = _qubit_xy(slm_map, current_placement[q])
        ax.scatter([x], [y], s=80, c=[_qubit_color(q)], zorder=8,
                   linewidths=0.8, edgecolors="white")
        xy = (x, y)
        idx = site_offsets[xy]
        site_offsets[xy] += 1
        if site_counts[xy] > 1:
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
    if is_entanglement_frame and layer_idx < n_layers:
        for pair in tq_layers[layer_idx]:
            q0, q1 = pair
            x0, y0 = _qubit_xy(slm_map, current_placement[q0])
            x1, y1 = _qubit_xy(slm_map, current_placement[q1])
            ax.plot([x0, x1], [y0, y1], "-", color="#c0392b",
                    linewidth=2.5, alpha=0.9, zorder=6)
            # CZ label at midpoint
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my - 2, "CZ", fontsize=4.5, ha="center", va="bottom",
                    color="#c0392b", fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white",
                              ec="#c0392b", alpha=0.85, lw=0.5))

    # ── movement arrows (storage/transition frames) ──────────────────────
    # Show BOTH directions:
    #   STORE arrows (dashed, faded) — how atoms arrived from previous ent. zone
    #   LOAD arrows (solid) — where atoms will move next
    if not is_entanglement_frame:
        # ── STORE arrows (incoming from previous entanglement) ───────────
        if frame_idx > 0 and (frame_idx - 1) < len(routing):
            prev_placement = placements[frame_idx - 1]
            store_qubits = [
                q for group in routing[frame_idx - 1] for q in group
                if _qubit_xy(slm_map, prev_placement[q]) != _qubit_xy(slm_map, current_placement[q])
            ]
            n_store = len(store_qubits)
            for rank, q in enumerate(store_qubits):
                x0, y0 = _qubit_xy(slm_map, prev_placement[q])
                x1, y1 = _qubit_xy(slm_map, current_placement[q])
                dist = math.hypot(x1 - x0, y1 - y0)
                rad = 0.0 if n_store == 1 else -0.25 + 0.5 * rank / max(n_store - 1, 1)
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color=_qubit_color(q),
                        lw=0.8, mutation_scale=8,
                        connectionstyle=f"arc3,rad={rad:.2f}",
                        linestyle="dashed", alpha=0.4,
                    ),
                    zorder=4,
                )
            # STORE direction label
            if store_qubits:
                ax.text(0.02, 0.02, f"\u2193 {n_store} stored ({_max_shuttle_dist(slm_map, prev_placement, current_placement, routing[frame_idx - 1]):.0f} \u00b5m)",
                        fontsize=6, transform=ax.transAxes, color="#2980b9",
                        fontweight="bold", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="#2980b9", alpha=0.8, lw=0.6))

        # ── LOAD arrows (outgoing to next entanglement) ─────────────────
        if frame_idx < len(routing):
            src = placements[frame_idx]
            dst = placements[frame_idx + 1]
            load_qubits = [
                q for group in routing[frame_idx] for q in group
                if _qubit_xy(slm_map, src[q]) != _qubit_xy(slm_map, dst[q])
            ]
            n_load = len(load_qubits)
            for rank, q in enumerate(load_qubits):
                x0, y0 = _qubit_xy(slm_map, src[q])
                x1, y1 = _qubit_xy(slm_map, dst[q])
                rad = 0.0 if n_load == 1 else -0.3 + 0.6 * rank / max(n_load - 1, 1)
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color=_qubit_color(q),
                        lw=1.2, mutation_scale=10,
                        connectionstyle=f"arc3,rad={rad:.2f}",
                    ),
                    zorder=5,
                )
            # LOAD direction label
            if load_qubits:
                ax.text(0.02, 0.06, f"\u2191 {n_load} loading ({_max_shuttle_dist(slm_map, src, dst, routing[frame_idx]):.0f} \u00b5m)",
                        fontsize=6, transform=ax.transAxes, color="#e67e22",
                        fontweight="bold", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="#e67e22", alpha=0.8, lw=0.6))

    # ── reuse markers ─────────────────────────────────────────────────────
    if is_entanglement_frame and layer_idx < len(reuse):
        for q in reuse[layer_idx]:
            x, y = _qubit_xy(slm_map, current_placement[q])
            ax.scatter([x], [y], s=200, facecolors="none",
                       edgecolors="#2ecc71", linewidths=1.5, zorder=10)

    # ── single-qubit gate indicators ──────────────────────────────────────
    # 1Q gates: tall vertical laser beams targeting the atom.
    from matplotlib.patches import Rectangle
    _, _, _, arch_ymax = _arch_axis_limits(arch)
    beam_color = "#00ffcc"
    core_color = "#e6ffff"

    sq_layers: list[list[dict[str, Any]]] = debug.get("single_qubit_layers", [])
    sq_on_qubit: dict[int, list[tuple[str, list[float]]]] = {}
    if not is_entanglement_frame and layer_idx < len(sq_layers):
        for op in sq_layers[layer_idx]:
            entry = (op["name"], op.get("params", []))
            for q in op["qubits"]:
                sq_on_qubit.setdefault(q, []).append(entry)

        MAX_LABELED = 20
        n_affected = len(sq_on_qubit)
        show_labels = n_affected <= MAX_LABELED

        for q_idx, (q, ops) in enumerate(sorted(sq_on_qubit.items())):
            x, y = _qubit_xy(slm_map, current_placement[q])

            # Draw laser beam from the top of the architecture down to the atom
            beam_width = 3.5
            beam_height = arch_ymax - y
            if beam_height < 0:
                beam_height = 20  # fallback

            ax.add_patch(Rectangle(
                (x - beam_width / 2, y), beam_width, beam_height,
                linewidth=0, facecolor=beam_color, alpha=0.25, zorder=4
            ))
            # Core of the beam
            ax.plot([x, x], [y, max(y, arch_ymax)], color=core_color, linewidth=1.2, alpha=0.9, zorder=5)

            # Impact glow on the atom
            ax.scatter([x], [y], s=500, facecolors=beam_color, alpha=0.35,
                       zorder=7, linewidths=0)
            ax.scatter([x], [y], s=260, facecolors="none",
                       edgecolors=beam_color, linewidths=1.5,
                       linestyles="dashed", zorder=10, alpha=0.9)

            if show_labels:
                parts = [_format_gate(name, params) for name, params in ops]
                # If multiple gates, stack them vertically
                label = "\n".join(parts)
                
                # Position the Quantikz-style box clearly visible on the atom/beam
                # We place it slightly above the atom exactly on the beam's path.
                y_label = y + 5
                
                ax.text(
                    x, y_label,
                    label,
                    fontsize=7, ha="center", va="center",
                    color="#111827", zorder=15,
                    bbox=dict(boxstyle="square,pad=0.4", fc="white",
                              ec="#9ca3af", linewidth=0.8, alpha=0.9)
                )

        if not show_labels:
            # Summarise
            gate_groups: dict[str, list[int]] = {}
            for q, ops in sq_on_qubit.items():
                sig = ", ".join(_format_gate(n, p) for n, p in ops)
                gate_groups.setdefault(sig, []).append(q)
            summary_parts = [
                f"{sig} \u00d7{len(qs)}" for sig, qs in gate_groups.items()
            ]
            summary = "1Q gates: " + ";  ".join(summary_parts)
            ax.text(0.5, 0.02, summary, fontsize=5.5, ha="center", va="bottom",
                    transform=ax.transAxes, color="#004d40", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#e0f2f1",
                              ec=beam_color, alpha=0.92, linewidth=0.8))

    # ── title ─────────────────────────────────────────────────────────────
    if is_entanglement_frame:
        cz_pairs = tq_layers[layer_idx] if layer_idx < n_layers else []
        n_cz = len(cz_pairs)
        pairs_str = ", ".join(f"(q{p[0]},q{p[1]})" for p in cz_pairs[:8])
        if len(cz_pairs) > 8:
            pairs_str += f" +{len(cz_pairs) - 8} more"
        ax.set_title(
            f"Layer {layer_idx}/{n_layers - 1}  \u2014  {n_cz} CZ gate{'s' if n_cz != 1 else ''}  "
            f"(Rydberg laser \u26a1)\n{pairs_str}\n"
            f"(frame {frame_idx}/{2 * n_layers})",
            fontsize=9)
    else:
        store_count = 0
        load_count = 0
        if frame_idx > 0 and (frame_idx - 1) < len(routing):
            store_count = sum(len(g) for g in routing[frame_idx - 1])
        if frame_idx < len(routing):
            load_count = sum(len(g) for g in routing[frame_idx])

        sq_note = ""
        if sq_on_qubit:
            gate_strs = []
            for q, ops in sorted(sq_on_qubit.items()):
                for name, params in ops:
                    gate_strs.append(f"{_format_gate(name, params)}\u2192q{q}")
            sq_note = "  |  1Q: " + ", ".join(gate_strs[:5])
            if len(gate_strs) > 5:
                sq_note += f" +{len(gate_strs) - 5}"

        move_parts = []
        if store_count:
            move_parts.append(f"\u2193{store_count} stored")
        if load_count:
            move_parts.append(f"\u2191{load_count} loading")
        move_str = ", ".join(move_parts) if move_parts else "idle"

        ax.set_title(
            f"Transition {layer_idx}  \u2014  {move_str}{sq_note}\n"
            f"(frame {frame_idx}/{2 * n_layers})", fontsize=9)

    return {
        "frame_idx": frame_idx,
        "layer_idx": layer_idx,
        "is_entanglement": is_entanglement_frame,
        "sq_on_qubit": sq_on_qubit,
    }


def _max_shuttle_dist(
    slm_map: dict[int, dict[str, Any]],
    src_placement: list[list[int]],
    dst_placement: list[list[int]],
    routing_groups: list[list[int]],
) -> float:
    """Return the maximum shuttle distance (μm) across all moving qubits."""
    max_d = 0.0
    for group in routing_groups:
        for q in group:
            x0, y0 = _qubit_xy(slm_map, src_placement[q])
            x1, y1 = _qubit_xy(slm_map, dst_placement[q])
            max_d = max(max_d, math.hypot(x1 - x0, y1 - y0))
    return max_d


# ── interpolated frame for smooth movie animation ────────────────────────────

def _render_interpolated_frame(
    ax: Any,
    arch: dict[str, Any],
    slm_map: dict[int, dict[str, Any]],
    debug: dict[str, Any],
    logical_frame: int,
    t: float,
) -> None:
    """Render a movement frame with atoms at interpolated positions.

    *t* ranges from 0.0 (at source placement) to 1.0 (at destination).
    Uses ease-in-out cubic interpolation for natural-looking shuttle movement.
    """
    n_qubits: int = debug["n_qubits"]
    placements = debug["placement"]
    routing = debug["routing"]
    n_layers: int = debug["n_layers"]

    # Ease-in-out cubic
    if t < 0.5:
        t_smooth = 4 * t * t * t
    else:
        t_smooth = 1 - (-2 * t + 2) ** 3 / 2

    src = placements[logical_frame]
    dst = placements[logical_frame + 1] if (logical_frame + 1) < len(placements) else src

    # Determine which qubits are moving
    moving: set[int] = set()
    if logical_frame < len(routing):
        for group in routing[logical_frame]:
            for q in group:
                if _qubit_xy(slm_map, src[q]) != _qubit_xy(slm_map, dst[q]):
                    moving.add(q)

    _draw_background(ax, arch, laser_active=False)

    # Draw atoms at interpolated positions
    for q in range(n_qubits):
        x0, y0 = _qubit_xy(slm_map, src[q])
        if q in moving:
            x1, y1 = _qubit_xy(slm_map, dst[q])
            x = x0 + (x1 - x0) * t_smooth
            y = y0 + (y1 - y0) * t_smooth
            # Trail from source
            ax.plot([x0, x], [y0, y], "-", color=_qubit_color(q),
                    linewidth=1.2, alpha=0.25, zorder=4)
            # Ghost at source
            if t_smooth < 0.9:
                ax.scatter([x0], [y0], s=40, c=[_qubit_color(q)],
                           alpha=0.15, zorder=3, linewidths=0)
            # Destination marker
            if t_smooth < 0.95:
                x1d, y1d = _qubit_xy(slm_map, dst[q])
                ax.scatter([x1d], [y1d], s=50, facecolors="none",
                           edgecolors=_qubit_color(q), linewidths=0.6,
                           alpha=0.3, zorder=3, linestyles="dotted")
        else:
            x, y = x0, y0

        ax.scatter([x], [y], s=80, c=[_qubit_color(q)], zorder=8,
                   linewidths=0.8, edgecolors="white")
        ax.text(x, y, str(q), fontsize=5, ha="center", va="center",
                color="white", fontweight="bold", zorder=9)

    # Title with progress
    layer_idx = logical_frame // 2
    n_moving = len(moving)
    direction = "\u2191 Loading" if (logical_frame % 2 == 0) else "\u2193 Storing"
    ax.set_title(
        f"Shuttling \u2014 {direction} \u2014 Layer {layer_idx} \u2014 "
        f"{n_moving} atom{'s' if n_moving != 1 else ''} "
        f"({t_smooth * 100:.0f}%)\n"
        f"(frame {logical_frame}/{2 * n_layers})",
        fontsize=9)


# ── circuit timeline & info panels ───────────────────────────────────────────

def _draw_circuit_timeline(
    ax: Any,
    debug: dict[str, Any],
    frame_idx: int,
) -> None:
    """Draw a compact layer timeline bar showing compilation progress.

    Each compilation layer is represented as a pair of coloured blocks:
    a yellow block for 1Q gates and a red block for 2Q CZ gates.  The
    current frame is marked with a bold vertical indicator.
    """
    from matplotlib.patches import Rectangle

    n_layers = debug["n_layers"]
    tq_layers = debug["two_qubit_layers"]
    sq_layers = debug.get("single_qubit_layers", [])
    total_frames = 2 * n_layers + 1

    # For large circuits show a sliding window of frames rather than all of them.
    WINDOW = 24
    if total_frames > WINDOW:
        half = WINDOW // 2
        w_start = max(0, min(frame_idx - half, total_frames - WINDOW))
        w_end = w_start + WINDOW
    else:
        w_start, w_end = 0, total_frames

    ax.set_xlim(w_start - 0.8, w_end - 0.2)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    visible_ticks = [f for f in range(w_start, w_end) if f % max(1, WINDOW // 12) == 0]
    ax.set_xticks(visible_ticks)
    ax.set_xticklabels([str(i) for i in visible_ticks], fontsize=4.5)
    ax.set_xlabel(
        f"Frame (showing {w_start}–{w_end - 1} of {total_frames - 1})"
        if total_frames > WINDOW else "Frame",
        fontsize=6, labelpad=2,
    )

    # Only draw blocks within the visible window (+ 1 past the edge for the final frame).
    for i in range(n_layers):
        sq_x = 2 * i
        tq_x = 2 * i + 1
        # Skip blocks entirely outside the visible window
        if tq_x < w_start or sq_x >= w_end:
            continue

        # 1Q gate block (even frame 2*i)
        n_sq = len(sq_layers[i]) if i < len(sq_layers) else 0
        fc_sq = "#f1c40f" if n_sq > 0 else "#f5f5f5"
        ec_sq = "#b7950b" if n_sq > 0 else "#cccccc"
        ax.add_patch(Rectangle(
            (sq_x - 0.4, -0.3), 0.8, 0.6,
            facecolor=fc_sq, edgecolor=ec_sq, linewidth=0.8,
            alpha=0.7, zorder=2))
        label_sq = f"{n_sq} 1Q" if n_sq else "\u2013"
        ax.text(sq_x, 0, label_sq, fontsize=4, ha="center", va="center",
                fontweight="bold", color="#7d6608" if n_sq else "#999999", zorder=3)

        # 2Q gate block (odd frame 2*i+1)
        n_tq = len(tq_layers[i])
        ax.add_patch(Rectangle(
            (tq_x - 0.4, -0.3), 0.8, 0.6,
            facecolor="#e74c3c", edgecolor="#c0392b", linewidth=0.8,
            alpha=0.7, zorder=2))
        ax.text(tq_x, 0, f"{n_tq} CZ", fontsize=4, ha="center", va="center",
                fontweight="bold", color="white", zorder=3)

    # Final storage frame (only draw if visible)
    final_x = 2 * n_layers
    if w_start <= final_x < w_end:
        ax.add_patch(Rectangle(
            (final_x - 0.4, -0.3), 0.8, 0.6,
            facecolor="#cde8ff", edgecolor="#2980b9", linewidth=0.8,
            alpha=0.7, zorder=2))
        ax.text(final_x, 0, "done", fontsize=4, ha="center", va="center",
                fontweight="bold", color="#2980b9", zorder=3)

    # Current frame indicator — always visible (inside window by construction)
    ax.axvline(x=frame_idx, color="#e74c3c", linewidth=2, zorder=5, alpha=0.8)
    ax.annotate(
        "\u25bc", xy=(frame_idx, -0.3),
        xytext=(frame_idx, -0.5),
        fontsize=8, ha="center", va="top", color="#e74c3c",
        fontweight="bold", zorder=6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _draw_frame_info(
    ax: Any,
    debug: dict[str, Any],
    frame_idx: int,
    arch: dict[str, Any],
    slm_map: dict[int, dict[str, Any]] | None = None,
    y_top: float = 0.95,
) -> None:
    """Draw a text panel showing detailed info about the current frame.

    *y_top* controls the vertical anchor (axes fraction) for the text block;
    set < 1 to leave room for a legend drawn above in the same axis.
    """
    n_layers = debug["n_layers"]
    tq_layers = debug["two_qubit_layers"]
    sq_layers = debug.get("single_qubit_layers", [])
    routing = debug["routing"]
    placements = debug["placement"]

    total_frames = 2 * n_layers + 1
    frame_idx = max(0, min(frame_idx, total_frames - 1))
    layer_idx = frame_idx // 2
    is_ent = (frame_idx % 2 == 1)

    if slm_map is None:
        slm_map = _build_slm_map(arch)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lines: list[str] = []

    if is_ent and layer_idx < n_layers:
        pairs = tq_layers[layer_idx]
        lines.append(f"\u25b6 2Q Gate Layer {layer_idx}: {len(pairs)} CZ pair{'s' if len(pairs) != 1 else ''}")
        pair_strs = [f"(q{p[0]}, q{p[1]})" for p in pairs]
        for i in range(0, len(pair_strs), 8):
            lines.append("    " + "  ".join(pair_strs[i:i + 8]))
        # Reuse info
        reuse = debug.get("reuse_qubits", [])
        if layer_idx < len(reuse) and reuse[layer_idx]:
            rq = ", ".join(f"q{q}" for q in reuse[layer_idx])
            lines.append(f"    \u267b Reused (stay in ent. zone): {rq}")
    elif not is_ent:
        # 1Q gate info
        if layer_idx < len(sq_layers) and sq_layers[layer_idx]:
            ops = sq_layers[layer_idx]
            lines.append(f"\u25b6 1Q Gate Layer {layer_idx}: {len(ops)} operation{'s' if len(ops) != 1 else ''}")
            for op in ops[:10]:
                gate = _format_gate(op["name"], op.get("params", []))
                qubits = ", ".join(f"q{q}" for q in op["qubits"])
                lines.append(f"    {gate} \u2192 {qubits}")
            if len(ops) > 10:
                lines.append(f"    \u2026 +{len(ops) - 10} more")
        else:
            lines.append(f"\u25b6 Transition {layer_idx}: no 1Q gates")

        # Movement info
        if frame_idx > 0 and (frame_idx - 1) < len(routing):
            groups = routing[frame_idx - 1]
            total = sum(len(g) for g in groups)
            if total > 0:
                dist = _max_shuttle_dist(slm_map, placements[frame_idx - 1],
                                         placements[frame_idx], groups)
                lines.append(f"    \u2193 Stored {total} atom{'s' if total != 1 else ''} "
                             f"in {len(groups)} group{'s' if len(groups) != 1 else ''} "
                             f"(max {dist:.1f} \u00b5m)")
        if frame_idx < len(routing):
            groups = routing[frame_idx]
            total = sum(len(g) for g in groups)
            if total > 0:
                dist = _max_shuttle_dist(slm_map, placements[frame_idx],
                                         placements[frame_idx + 1], groups)
                lines.append(f"    \u2191 Loading {total} atom{'s' if total != 1 else ''} "
                             f"in {len(groups)} group{'s' if len(groups) != 1 else ''} "
                             f"(max {dist:.1f} \u00b5m)")

    if not lines:
        lines.append(f"\u25b6 Frame {frame_idx}: Final storage placement")

    # Timing estimate
    op_dur = arch.get("operation_duration", {})
    if op_dur:
        t_transfer = op_dur.get("atom_transfer", 15.0)
        t_cz = op_dur.get("rydberg_gate", 0.36)
        t_1q = op_dur.get("single_qubit_gate", 0.625)
        if is_ent:
            lines.append(f"    ~ Est. duration: {t_cz:.2f} \u00b5s (CZ gate)")
        else:
            t_est = 0.0
            if layer_idx < len(sq_layers) and sq_layers[layer_idx]:
                qc: dict[int, int] = defaultdict(int)
                for op in sq_layers[layer_idx]:
                    for q in op["qubits"]:
                        qc[q] += 1
                t_est += t_1q * (max(qc.values()) if qc else 0)
            # Shuttle time estimate: t_transfer per load/store
            if frame_idx < len(routing):
                t_est += t_transfer
            if frame_idx > 0 and (frame_idx - 1) < len(routing):
                t_est += t_transfer
            if t_est > 0:
                lines.append(f"    ~ Est. duration: {t_est:.2f} \u00b5s")

    text = "\n".join(lines)
    ax.text(0.01, y_top, text, fontsize=5.5, va="top", ha="left",
            family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa",
                      ec="#dee2e6", alpha=0.9, linewidth=0.5))




# ── legend construction ──────────────────────────────────────────────────────

def _build_legends(
    ax: Any,
    debug: dict[str, Any],
    is_entanglement: bool,
) -> None:
    """Add qubit-colour and visual-element legends to a dedicated axis.

    The axis should be sized appropriately in the caller's GridSpec.
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_q = debug["n_qubits"]

    qubit_handles = [
        mpatches.Patch(color=_qubit_color(q), label=f"q{q}")
        for q in range(n_q)
    ]
    element_handles = [
        mpatches.Patch(facecolor="#cde8ff", edgecolor="#2980b9",
                       label="Storage zone"),
        mpatches.Patch(facecolor="#fde8c8", edgecolor="#e67e22",
                       label="Entanglement zone"),
        mpatches.Patch(facecolor="#ede0f5", edgecolor="#7d3c98",
                       linestyle="dashed", label="Rydberg laser range"),
        mlines.Line2D([], [], color="none", marker="o", markersize=8,
                      markerfacecolor="#4a90d9", markeredgecolor="white",
                      label="Atom (colour = qubit index)"),
    ]
    if is_entanglement:
        element_handles += [
            mpatches.Patch(facecolor="#ff6b6b", edgecolor="#e74c3c",
                           alpha=0.7, label="Rydberg laser firing"),
            mlines.Line2D([], [], color="#c0392b", linewidth=2,
                          label="CZ gate bond"),
            mlines.Line2D([], [], color="none", marker="o", markersize=10,
                          markerfacecolor="none", markeredgecolor="#2ecc71",
                          markeredgewidth=1.5, label="Reused qubit (stays in ent. zone)"),
        ]
    else:
        element_handles += [
            mlines.Line2D([], [], color="#888888", linewidth=1.5,
                          marker=">", markersize=5,
                          label="LOAD arrow (\u2192 ent. zone)"),
            mlines.Line2D([], [], color="#888888", linewidth=0.8,
                          marker=">", markersize=4, linestyle="dashed",
                          alpha=0.5, label="STORE arrow (\u2190 ent. zone)"),
            mlines.Line2D([], [], color="none", marker="o", markersize=8,
                          markerfacecolor="none", markeredgecolor="#f1c40f",
                          markeredgewidth=1.5, linestyle="dashed",
                          label="1Q gate being applied (dashed ring)"),
        ]

    # Qubit legend: compact when many qubits (smaller patches, more columns).
    MAX_LABELED_Q = 32
    if n_q <= MAX_LABELED_Q:
        q_fontsize = max(3.5, 6.0 - n_q * 0.06)
        q_ncol = max(1, (n_q + 15) // 16)
        q_handles = qubit_handles
    else:
        # Strip labels for very large qubit counts — show colour swatches only.
        q_fontsize = 3.5
        q_ncol = max(1, (n_q + 23) // 24)
        import matplotlib.patches as _mp2
        q_handles = [_mp2.Patch(color=_qubit_color(q), label="") for q in range(n_q)]

    qubit_leg = ax.legend(
        handles=q_handles,
        loc="upper left", bbox_to_anchor=(0.0, 1.0),
        fontsize=q_fontsize,
        ncol=q_ncol,
        framealpha=0.9,
        title=f"Qubits ({n_q})",
        title_fontsize=6,
        borderaxespad=0.3,
        handlelength=0.8, handleheight=0.6, handletextpad=0.3,
        columnspacing=0.5,
    )
    ax.add_artist(qubit_leg)
    ax.legend(
        handles=element_handles,
        loc="lower left", bbox_to_anchor=(0.0, 0.0),
        fontsize=5.5,
        framealpha=0.9,
        title="Visual elements",
        title_fontsize=6,
        borderaxespad=0.3,
    )


# ── circuit schedule overview ─────────────────────────────────────────────────

def _draw_circuit_overview(
    ax: Any,
    debug: dict[str, Any],
    frame_idx: int,
) -> None:
    """Draw the transformed circuit in a Quantikz style."""
    from matplotlib.patches import Rectangle
    
    n_layers = debug["n_layers"]
    n_qubits = debug["n_qubits"]
    tq_layers = debug["two_qubit_layers"]
    sq_layers = debug.get("single_qubit_layers", [])
    reuse = debug.get("reuse_qubits", [])
    
    layer_idx = frame_idx // 2
    
    ax.cla()
    if n_layers == 0 or n_qubits == 0:
        ax.axis("off")
        return
        
    MAX_L = 20
    if n_layers > MAX_L:
        half = MAX_L // 2
        l_start = max(0, min(layer_idx - half, n_layers - MAX_L))
        l_end = l_start + MAX_L
    else:
        l_start, l_end = 0, n_layers
        
    MAX_Q = 20
    # To keep it centered around active qubits, find active set
    active_qubits: set[int] = set()
    if layer_idx < len(tq_layers):
        for p1, p2 in tq_layers[layer_idx]:
            active_qubits.update([p1, p2])
    if layer_idx < len(sq_layers) and sq_layers[layer_idx]:
        for op in sq_layers[layer_idx]:
            active_qubits.update(op["qubits"])
            
    avg_q = sum(active_qubits) // len(active_qubits) if active_qubits else n_qubits // 2
    if n_qubits > MAX_Q:
        q_start = max(0, min(avg_q - MAX_Q // 2, n_qubits - MAX_Q))
        q_end = q_start + MAX_Q
    else:
        q_start, q_end = 0, n_qubits

    n_vis_layers = l_end - l_start
    n_vis_qubits = q_end - q_start

    # Coordinate system: x from 0 to n_vis_layers, y from 0 to n_vis_qubits
    ax.set_xlim(-1.5, n_vis_layers + 0.5)
    ax.set_ylim(-0.8, n_vis_qubits + 0.8)
    ax.invert_yaxis()  # Qubit 0 at the top
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw layer labels
    for vi, l in enumerate(range(l_start, l_end)):
        ax.text(vi, -0.6, f"L{l}", fontsize=6, ha="center", va="bottom", color="#444444", fontweight="bold")

    # Draw Qubit wire lines and labels
    for vi, q in enumerate(range(q_start, q_end)):
        ax.plot([-0.5, n_vis_layers - 0.5], [vi, vi], color="black", linewidth=1.0, zorder=1)
        ax.text(-0.7, vi, f"q{q}", fontsize=7, ha="right", va="center", color="black", fontweight="bold", zorder=2)
        
    # Draw Gates
    for vi, l in enumerate(range(l_start, l_end)):
        is_cur = (l == layer_idx)
        
        # Current layer highlight
        if is_cur:
            c_hl = "#f4d03f" if frame_idx % 2 == 0 else "#e74c3c"
            ax.add_patch(Rectangle(
                (vi - 0.4, -0.4),
                0.8, n_vis_qubits - 0.2,
                facecolor=c_hl, edgecolor=c_hl, alpha=0.15, zorder=0, linewidth=1.5,
                linestyle="dashed"
            ))
            ax.add_patch(Rectangle(
                (vi - 0.4, -0.4),
                0.8, n_vis_qubits - 0.2,
                facecolor="none", edgecolor=c_hl, alpha=0.8, zorder=0, linewidth=2
            ))
            
        # Draw 1Q blocks
        if l < len(sq_layers) and sq_layers[l]:
            for op in sq_layers[l]:
                for q in op["qubits"]:
                    if q_start <= q < q_end:
                        yq = q - q_start
                        gate_label = _format_gate(op["name"], op.get("params", []))
                        # Draw a cute square with text
                        ax.text(vi, yq, gate_label, fontsize=6.5, ha="center", va="center",
                                color="black", zorder=10,
                                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1.2))
                                
        # Draw 2Q blocks (CZ)
        if l < len(tq_layers) and tq_layers[l]:
            for p1, p2 in tq_layers[l]:
                if q_start <= p1 < q_end or q_start <= p2 < q_end:
                    v1 = max(0, min(p1 - q_start, n_vis_qubits - 1))
                    v2 = max(0, min(p2 - q_start, n_vis_qubits - 1))
                    
                    # Vertical connector
                    ax.plot([vi, vi], [min(v1, v2), max(v1, v2)], color="black", linewidth=1.5, zorder=5)
                    # Dots
                    if q_start <= p1 < q_end:
                        ax.scatter([vi], [p1 - q_start], s=40, color="black", zorder=6)
                    if q_start <= p2 < q_end:
                        ax.scatter([vi], [p2 - q_start], s=40, color="black", zorder=6)
                        
    suffix = f" (Showing L{l_start}-L{l_end-1})" if n_layers > MAX_L else ""
    ax.set_title(f"Transformed Circuit {suffix}", fontsize=9, pad=8, fontweight="bold")


# ── public API ─────────────────────────────────────────────────────────────────


def visualize_compilation_step(
    debug: dict[str, Any],
    arch: str | dict[str, Any],
    frame: int = 0,
    figsize: tuple[float, float] | None = None,
    show_circuit: bool = True,
) -> "matplotlib.figure.Figure":
    """Render a single frame of the compilation animation.

    Each *frame* alternates between two views:

    * **Even frames** (0, 2, 4, …): atoms are in their storage placement
      with movement arrows showing both incoming STORE (dashed, faded) and
      outgoing LOAD (solid) shuttle paths. 1Q gates are shown as focused
      laser beams with gate names and rotation angles.
    * **Odd frames** (1, 3, 5, …): atoms have arrived at the entanglement
      zone; active CZ pairs are highlighted in red. Reused qubits (those
      that stay in the entanglement zone for the next layer) are circled
      in green.

    The total number of frames is ``2 * n_layers + 1``.

    When *show_circuit* is True, a circuit timeline bar and detailed info
    panel are rendered below the main atom view.

    Args:
        debug: The dictionary returned by ``compiler.debug_info()``.
        arch: Architecture JSON string or dict.
        frame: Frame index to render (0-based).
        figsize: Figure size in inches.
        show_circuit: If True, show circuit timeline and info panels.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as e:
        msg = "matplotlib is required: pip install matplotlib"
        raise ImportError(msg) from e

    arch_data: dict[str, Any] = json.loads(arch) if isinstance(arch, str) else dict(arch)
    slm_map = _build_slm_map(arch_data)
    n_layers = debug["n_layers"]
    total_frames = 2 * n_layers + 1
    frame = max(0, min(frame, total_frames - 1))

    # Layout: 2-column GridSpec.
    #   Left column (78%):  atom view | gate schedule overview | timeline
    #   Right column (22%): legends + info panel
    if show_circuit and n_layers > 0:
        fig = plt.figure(figsize=figsize or (19.2, 10.8))
        gs = GridSpec(
            3, 2, figure=fig,
            height_ratios=[5.5, 1.6, 0.7],
            width_ratios=[4.5, 1],
            left=0.05, right=0.98, top=0.93, bottom=0.06,
            wspace=0.04, hspace=0.35,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_overview = fig.add_subplot(gs[1, 0])
        ax_timeline = fig.add_subplot(gs[2, 0])
        ax_right = fig.add_subplot(gs[:, 1])   # full right column: legends + info
    else:
        fig, ax = plt.subplots(figsize=figsize or (14, 9))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07)
        ax_overview = None
        ax_timeline = None
        ax_right = None

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("y (\u00b5m)")
    ax.grid(visible=True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    xmin, xmax, ymin, ymax = _arch_axis_limits(arch_data)

    _render_frame(ax, arch_data, slm_map, debug, frame)

    # Enforce fixed limits AFTER render so set_aspect("box") shrinks the box,
    # not the data limits — prevents axis jumping between frames.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)   # ymax first → y-axis inverted (y increases downward)

    is_ent = (frame % 2 == 1) and (frame // 2 < n_layers)
    if ax_right is not None:
        _build_legends(ax_right, debug, is_ent)
        _draw_frame_info(ax_right, debug, frame, arch_data, slm_map,
                         y_top=0.48)

    if ax_overview is not None:
        _draw_circuit_overview(ax_overview, debug, frame)
    if ax_timeline is not None:
        _draw_circuit_timeline(ax_timeline, debug, frame)

    return fig


def _has_movement_between(
    slm_map: dict[int, dict[str, Any]],
    placements: list[list[list[int]]],
    routing: list[list[list[int]]],
    logical_frame: int,
    direction: str,
) -> bool:
    """Return True if any qubit moves in *direction* at *logical_frame*.

    ``direction="store"`` checks movement from the previous entanglement
    placement back to storage; ``"load"`` checks forward to the next
    entanglement placement.
    """
    if direction == "store":
        ri, src_idx, dst_idx = logical_frame - 1, logical_frame - 1, logical_frame
    else:
        ri, src_idx, dst_idx = logical_frame, logical_frame, logical_frame + 1
    if ri < 0 or ri >= len(routing):
        return False
    if src_idx >= len(placements) or dst_idx >= len(placements):
        return False
    return any(
        _qubit_xy(slm_map, placements[src_idx][q]) != _qubit_xy(slm_map, placements[dst_idx][q])
        for group in routing[ri] for q in group
    )


def _build_expanded_frames(
    debug: dict[str, Any],
    slm_map: dict[int, dict[str, Any]],
    n_logical: int,
    smooth_movement: bool,
    sub_frames_per_move: int,
    sub_frames_per_gate: int = 10,
) -> tuple[list[Any], bool]:
    """Build the expanded frame list shared by the animation and save helpers.

    When the *debug* dict contains an ``"operations"`` key (ops-mode) and
    *smooth_movement* is True, each load/move/store operation produces one or
    more entries so that atoms are animated step-by-step.  Otherwise, a
    simpler per-routing-group interpolation is used.

    Returns:
        ``(expanded, use_ops)`` where *use_ops* is True when the ops path is
        taken.  Items in *expanded* are dicts (ops path) or tuples
        ``(logical_frame, phase, t, group_idx)`` (legacy path).
    """
    routing = debug["routing"]
    placements = debug["placement"]
    use_ops = "operations" in debug
    expanded: list[Any] = []

    if use_ops and smooth_movement:
        ops = debug["operations"]
        n_qubits = debug.get("n_qubits", 0)
        current_pos: dict[int, tuple[float, float]] = {}
        if placements:
            for q in range(n_qubits):
                current_pos[q] = _qubit_xy(slm_map, placements[0][q])
        logical_frame = 0
        active_load: set[int] = set()
        for op in ops:
            if logical_frame >= n_logical:
                break
            op_type = op.get("type", "unknown")
            qubits = op.get("qubits", [])
            if op_type == "load":
                for q in qubits:
                    active_load.add(q)
                expanded.append({
                    "phase": "load",
                    "logical_frame": logical_frame,
                    "current_pos": dict(current_pos),
                    "active_qubits": list(qubits),
                    "active_load": set(active_load),
                    "t": 1.0,
                })
            elif op_type == "store":
                for q in qubits:
                    active_load.discard(q)
                expanded.append({
                    "phase": "store",
                    "logical_frame": logical_frame,
                    "current_pos": dict(current_pos),
                    "active_qubits": list(qubits),
                    "active_load": set(active_load),
                    "t": 1.0,
                })
            elif op_type == "move":
                targets = op.get("targets", [])
                start_pos = {q: current_pos[q] for q in qubits}
                end_pos = {q: (targets[i][0], targets[i][1]) for i, q in enumerate(qubits)}
                for s in range(sub_frames_per_move):
                    t = s / max(sub_frames_per_move - 1, 1)
                    t_smooth = 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2)**3 / 2
                    frame_pos = dict(current_pos)
                    for q in qubits:
                        x0, y0 = start_pos[q]
                        x1, y1 = end_pos[q]
                        frame_pos[q] = (x0 + (x1 - x0) * t_smooth, y0 + (y1 - y0) * t_smooth)
                    expanded.append({
                        "phase": "move",
                        "logical_frame": logical_frame,
                        "current_pos": frame_pos,
                        "active_qubits": list(qubits),
                        "active_load": set(active_load),
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "t": t_smooth,
                    })
                for q in qubits:
                    current_pos[q] = end_pos[q]
            elif op_type == "cz":
                if logical_frame % 2 == 0:
                    logical_frame += 1
                for s in range(sub_frames_per_gate):
                    t = s / max(sub_frames_per_gate - 1, 1)
                    expanded.append({
                        "phase": "cz",
                        "logical_frame": logical_frame,
                        "current_pos": dict(current_pos),
                        "active_qubits": list(qubits),
                        "active_load": set(active_load),
                        "t": t,
                    })
                logical_frame += 1
            elif op_type in ("local_u", "local_rz"):
                for s in range(sub_frames_per_gate):
                    t = s / max(sub_frames_per_gate - 1, 1)
                    expanded.append({
                        "phase": "1q",
                        "logical_frame": logical_frame,
                        "current_pos": dict(current_pos),
                        "active_qubits": list(qubits),
                        "active_load": set(active_load),
                        "t": t,
                        "gate_name": op.get("name", op_type),
                    })
    elif smooth_movement:
        for f in range(n_logical):
            is_ent = (f % 2 == 1)
            if is_ent:
                expanded.append((f, "static", 1.0, 0))
            else:
                if _has_movement_between(slm_map, placements, routing, f, "store"):
                    for g_idx in range(len(routing[f - 1])):
                        for s in range(sub_frames_per_move):
                            t = s / max(sub_frames_per_move - 1, 1)
                            expanded.append((f, "store_move", t, g_idx))
                expanded.append((f, "static", 1.0, 0))
                if _has_movement_between(slm_map, placements, routing, f, "load"):
                    for g_idx in range(len(routing[f])):
                        for s in range(sub_frames_per_move):
                            t = s / max(sub_frames_per_move - 1, 1)
                            expanded.append((f, "load_move", t, g_idx))
    else:
        for f in range(n_logical):
            expanded.append((f, "static", 1.0, 0))

    return expanded, use_ops


def animate_compilation(
    debug: dict[str, Any],
    arch: str | dict[str, Any],
    interval: int = 800,
    figsize: tuple[float, float] | None = None,
    repeat: bool = True,
    max_frames: int | None = None,
    show_circuit: bool = True,
    smooth_movement: bool = True,
    sub_frames_per_move: int = 6,
    sub_frames_per_gate: int = 8,
) -> "matplotlib.animation.FuncAnimation":
    """Create an animated step-through of the full compilation.

    Each logical frame alternates between storage placements (with LOAD/STORE
    arrows) and entanglement-zone placements (with CZ highlights).

    When *smooth_movement* is True (default), qubit shuttling is animated with
    ``sub_frames_per_move`` interpolated sub-frames between logical frames.
    The STORE phase (entanglement → storage) and the LOAD phase (storage →
    entanglement) are shown as separate smooth movement sequences so that
    the route chosen by the compiler in both directions is clearly visible.
    Qubits belonging to the same routing group (moved together in the AOD)
    are connected with a thin line during movement.

    When *show_circuit* is True, a circuit timeline and info panel are
    displayed below the main view.

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
        interval: Milliseconds between static/gate frames.
        figsize: Figure size in inches.
        repeat: Whether the animation loops.
        max_frames: Cap the number of *logical* frames rendered (useful for
                    large circuits). ``None`` renders all frames.
        show_circuit: If True, show circuit timeline and info panels.
        smooth_movement: If True, insert interpolated sub-frames for each
                         shuttle movement so atoms glide rather than teleport.
        sub_frames_per_move: Number of sub-frames per shuttle direction when
                             smooth_movement is True.
        sub_frames_per_gate: Number of pulse sub-frames for 1Q/2Q effects
                             in ops-based animation mode.

    Returns:
        A :class:`matplotlib.animation.FuncAnimation`.
    """
    try:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as e:
        msg = "matplotlib is required: pip install matplotlib"
        raise ImportError(msg) from e

    arch_data: dict[str, Any] = json.loads(arch) if isinstance(arch, str) else dict(arch)
    slm_map = _build_slm_map(arch_data)
    n_layers = debug["n_layers"]

    n_logical = 2 * n_layers + 1
    if max_frames is not None:
        n_logical = min(n_logical, max_frames)

    expanded, use_ops = _build_expanded_frames(
        debug,
        slm_map,
        n_logical,
        smooth_movement,
        sub_frames_per_move,
        sub_frames_per_gate,
    )
    n_expanded = len(expanded)

    if show_circuit and n_layers > 0:
        fig = plt.figure(figsize=figsize or (19.2, 10.8))
        gs = GridSpec(
            3, 2, figure=fig,
            height_ratios=[5.5, 1.6, 0.7],
            width_ratios=[4.5, 1],
            left=0.05, right=0.98, top=0.93, bottom=0.06,
            wspace=0.04, hspace=0.35,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_overview = fig.add_subplot(gs[1, 0])
        ax_timeline = fig.add_subplot(gs[2, 0])
        ax_right = fig.add_subplot(gs[:, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize or (14, 9))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07)
        ax_overview = None
        ax_timeline = None
        ax_right = None

    xmin, xmax, ymin, ymax = _arch_axis_limits(arch_data)

    # Per-expanded-frame interval: movement sub-frames are faster.
    move_interval = max(25, interval // max(sub_frames_per_move, 1))
    gate_interval = max(25, interval // max(sub_frames_per_gate, 1))
    intervals = []
    for item in expanded:
        phase = item["phase"] if use_ops else item[1]
        if phase in ("store_move", "load_move", "move", "load", "store"):
            intervals.append(move_interval)
        elif phase in ("cz", "1q"):
            intervals.append(gate_interval)
        else:
            intervals.append(interval)
    first_interval = intervals[0] if intervals else interval
    last_panel_frame = -1
    anim: Any | None = None

    def _update(exp_idx: int) -> None:
        nonlocal last_panel_frame
        if use_ops and smooth_movement:
            fdat = expanded[exp_idx]
            logical_frame = fdat["logical_frame"]
            phase = fdat["phase"]
            t = fdat.get("t", 1.0)
            group_idx = 0
        else:
            logical_frame, phase, t, group_idx = expanded[exp_idx]

        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("")
        ax.set_ylabel("y (\u00b5m)")
        ax.grid(visible=True, linestyle=":", linewidth=0.4,
                color="#cccccc", zorder=0)

        if use_ops and smooth_movement and phase in ("move", "load", "store", "cz", "1q"):
            _render_ops_frame(ax, arch_data, slm_map, debug, fdat)
        elif phase == "store_move":
            _render_directional_interpolation(
                ax, arch_data, slm_map, debug, logical_frame,
                direction="store", t=t, group_idx=group_idx)
        elif phase == "load_move":
            _render_directional_interpolation(
                ax, arch_data, slm_map, debug, logical_frame,
                direction="load", t=t, group_idx=group_idx)
        else:
            _render_frame(ax, arch_data, slm_map, debug, logical_frame)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)

        is_ent = (logical_frame % 2 == 1) and (logical_frame // 2 < n_layers)
        if logical_frame != last_panel_frame:
            if ax_right is not None:
                ax_right.cla()
                _build_legends(ax_right, debug, is_ent)
                _draw_frame_info(ax_right, debug, logical_frame, arch_data, slm_map,
                                 y_top=0.48)

            if ax_overview is not None:
                ax_overview.cla()
                _draw_circuit_overview(ax_overview, debug, logical_frame)
            if ax_timeline is not None:
                ax_timeline.cla()
                _draw_circuit_timeline(ax_timeline, debug, logical_frame)
            last_panel_frame = logical_frame

        if anim is not None and (exp_idx + 1) < len(intervals):
            anim.event_source.interval = intervals[exp_idx + 1]

    anim = animation.FuncAnimation(fig, _update, frames=n_expanded,
                                   interval=first_interval, repeat=repeat,
                                   blit=False)
    return anim


def save_compilation_animation(
    debug: dict[str, Any],
    arch: str | dict[str, Any],
    path: str,
    fps: float = 4.0,
    figsize: tuple[float, float] | None = None,
    dpi: int = 80,
    max_frames: int | None = None,
    start_frame: int = 0,
    show_circuit: bool = True,
    smooth_movement: bool = True,
    sub_frames_per_move: int = 4,
    sub_frames_per_gate: int = 50,
    gate_duration_s: float | None = 1.5,
    verbose: bool = True,
) -> None:
    """Save a compilation animation directly to a GIF or MP4 file.

    This is the **memory-efficient** alternative to calling
    ``animate_compilation(...).save(...)``.  Each frame is rendered into a
    temporary PNG on disk, the matplotlib figure is closed immediately, and
    only the tiny PNG file remains until final assembly — so peak RAM usage
    is independent of the total frame count.

    When *smooth_movement* is True (default) and *debug* contains an
    ``"operations"`` key, every load/move/store step is expanded into
    *sub_frames_per_move* intermediate frames so atoms glide smoothly,
    matching the visual detail of :func:`animate_compilation`.

    GIF output requires ``pillow``; MP4/WebM output requires ``ffmpeg``.
    High-quality GIF also benefits from ``ffmpeg`` (via palette optimisation);
    pass ``use_ffmpeg_gif=True`` to force it when ffmpeg is available.

    Recommended parameters for large circuits:
      - ``figsize=(12, 6.75)`` + ``dpi=80``  →  960 × 540 px per frame
      - ``sub_frames_per_move=4``             →  fewer frames, faster render
      - ``max_frames=50``                     →  cap during exploration

    Args:
        debug: The dictionary returned by ``compiler.debug_info()``.
        arch: Architecture JSON string or dict.
        path: Output file path.  Extension determines format
              (``.gif`` / ``.mp4`` / ``.webm``).
        fps: Frames per second for static/gate frames.  Movement sub-frames
             are played at ``fps * sub_frames_per_move`` so they feel fluid.
        figsize: Figure size in inches.  Defaults to ``(12, 6.75)``.
        dpi: Rasterisation dots-per-inch.  Lower values reduce file size.
        max_frames: Cap on logical frames (2·n_layers+1 total).  Does not
                    count movement sub-frames.
        start_frame: First logical frame to include (0-based).  Frames before
                     this index are skipped so the animation begins mid-circuit.
                     The circuit panel still reflects the full debug context.
        show_circuit: Show circuit timeline and info panel.
        smooth_movement: If True, build full ops-based sub-frame sequence.
        sub_frames_per_move: Interpolated frames per atom movement.
        verbose: Print progress every 10 frames.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as e:
        msg = "matplotlib is required: pip install matplotlib"
        raise ImportError(msg) from e

    arch_data: dict[str, Any] = json.loads(arch) if isinstance(arch, str) else dict(arch)
    slm_map = _build_slm_map(arch_data)
    n_layers = debug["n_layers"]
    n_logical = 2 * n_layers + 1
    if max_frames is not None:
        n_logical = min(n_logical, max_frames)

    xmin, xmax, ymin, ymax = _arch_axis_limits(arch_data)
    _figsize = figsize or (12.0, 6.75)

    expanded, use_ops = _build_expanded_frames(
        debug, slm_map, n_logical, smooth_movement, sub_frames_per_move, sub_frames_per_gate
    )

    # Drop expanded entries whose logical frame precedes start_frame
    if start_frame > 0:
        def _lf(item: Any) -> int:
            return item["logical_frame"] if use_ops else item[0]
        expanded = [e for e in expanded if _lf(e) >= start_frame]

    n_expanded = len(expanded)

    # Movement sub-frames play faster so shuttling looks smooth
    move_fps = fps * max(sub_frames_per_move, 1)

    def _render_item_to_fig(item: Any) -> "matplotlib.figure.Figure":
        """Create the full figure layout, render one expanded frame, return it."""
        if use_ops and smooth_movement:
            fdat = item
            logical_frame: int = fdat["logical_frame"]
            phase: str = fdat["phase"]
            t: float = fdat.get("t", 1.0)
            group_idx: int = 0
        else:
            logical_frame, phase, t, group_idx = item

        is_ent = (logical_frame % 2 == 1) and (logical_frame // 2 < n_layers)

        if show_circuit and n_layers > 0:
            fig = plt.figure(figsize=_figsize)
            gs = GridSpec(
                3, 2, figure=fig,
                height_ratios=[5.5, 1.6, 0.7],
                width_ratios=[4.5, 1],
                left=0.05, right=0.98, top=0.93, bottom=0.06,
                wspace=0.04, hspace=0.35,
            )
            ax = fig.add_subplot(gs[0, 0])
            ax_overview = fig.add_subplot(gs[1, 0])
            ax_timeline = fig.add_subplot(gs[2, 0])
            ax_right = fig.add_subplot(gs[:, 1])
        else:
            fig, ax = plt.subplots(figsize=_figsize)
            fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07)
            ax_overview = ax_timeline = ax_right = None

        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("y (\u00b5m)")
        ax.grid(visible=True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

        if use_ops and smooth_movement and phase in ("move", "load", "store", "cz", "1q"):
            _render_ops_frame(ax, arch_data, slm_map, debug, fdat)
        elif not (use_ops and smooth_movement) and phase == "store_move":
            _render_directional_interpolation(
                ax, arch_data, slm_map, debug, logical_frame,
                direction="store", t=t, group_idx=group_idx)
        elif not (use_ops and smooth_movement) and phase == "load_move":
            _render_directional_interpolation(
                ax, arch_data, slm_map, debug, logical_frame,
                direction="load", t=t, group_idx=group_idx)
        else:
            _render_frame(ax, arch_data, slm_map, debug, logical_frame)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)

        if ax_right is not None:
            _build_legends(ax_right, debug, is_ent)
            _draw_frame_info(ax_right, debug, logical_frame, arch_data, slm_map, y_top=0.48)
        if ax_overview is not None:
            _draw_circuit_overview(ax_overview, debug, logical_frame)
        if ax_timeline is not None:
            _draw_circuit_timeline(ax_timeline, debug, logical_frame)

        return fig

    # ── Pre-compute per-frame durations ────────────────────────────────────
    # Each gate sub-frame duration is chosen so the full gate animation lasts
    # exactly gate_duration_s seconds (when specified), regardless of fps.
    if gate_duration_s is not None:
        _gate_frame_ms = max(20, int(gate_duration_s * 1000 / max(sub_frames_per_gate, 1)))
    else:
        _gate_frame_ms = max(20, int(1000 / max(fps * 0.2, 0.3)))

    frame_durations_ms: list[int] = []
    for item in expanded:
        _phase = item["phase"] if (use_ops and smooth_movement) else item[1]
        _is_move = _phase in ("move", "store_move", "load_move")
        _is_gate = _phase in ("cz", "1q")
        frame_durations_ms.append(_gate_frame_ms if _is_gate else
                                  int(1000 / move_fps) if _is_move else
                                  int(1000 / fps))

    # ── Render all frames to a temporary directory ─────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="qmap_anim_")
    frame_paths: list[str] = []

    try:
        for i, item in enumerate(expanded):
            fig = _render_item_to_fig(item)
            fpath = os.path.join(tmpdir, f"frame_{i:05d}.png")
            fig.savefig(fpath, dpi=dpi)
            plt.close(fig)   # ← release matplotlib memory immediately
            frame_paths.append(fpath)

            if verbose and (i + 1) % 10 == 0:
                print(f"  rendered {i + 1}/{n_expanded} frames...")

        # ── Assemble output ────────────────────────────────────────────────
        path_lower = path.lower()
        if path_lower.endswith(".mp4") or path_lower.endswith(".webm"):
            # Variable-duration MP4: use ffmpeg concat demuxer
            concat_file = os.path.join(tmpdir, "concat.txt")
            with open(concat_file, "w") as f:
                for fpath, dur_ms in zip(frame_paths, frame_durations_ms):
                    f.write(f"file '{fpath}'\n")
                    f.write(f"duration {dur_ms / 1000:.6f}\n")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                path,
            ]
            subprocess.run(cmd, check=True, capture_output=not verbose)

        elif path_lower.endswith(".gif"):
            # Try ffmpeg first (better palette → smaller, higher quality GIF)
            ffmpeg_ok = False
            try:
                concat_file = os.path.join(tmpdir, "concat.txt")
                with open(concat_file, "w") as f:
                    for fpath, dur_ms in zip(frame_paths, frame_durations_ms):
                        f.write(f"file '{fpath}'\n")
                        f.write(f"duration {dur_ms / 1000:.6f}\n")
                palette_path = os.path.join(tmpdir, "palette.png")
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", concat_file,
                     "-vf", "palettegen=stats_mode=diff", palette_path],
                    check=True, capture_output=True,
                )
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", concat_file, "-i", palette_path,
                     "-filter_complex", "paletteuse=dither=bayer",
                     "-loop", "0",
                     path],
                    check=True, capture_output=True,
                )
                ffmpeg_ok = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            if not ffmpeg_ok:
                # Fallback: accumulate PIL images (each is much smaller than a figure)
                try:
                    from PIL import Image
                except ImportError as e:
                    msg = "pillow is required for GIF output: pip install pillow"
                    raise ImportError(msg) from e
                pil_frames = [Image.open(fp).copy() for fp in frame_paths]
                pil_frames[0].save(
                    path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=frame_durations_ms,
                    loop=0,
                    optimize=True,
                )
        else:
            msg = f"Unsupported output format: {path!r} (expected .gif, .mp4, or .webm)"
            raise ValueError(msg)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if verbose:
        size_mb = os.path.getsize(path) / 1e6
        print(f"Saved {path}  ({n_expanded} frames, {size_mb:.1f} MB)")


# ── movie mode (time-proportional animation) ─────────────────────────────────

def _compute_frame_times(
    debug: dict[str, Any],
    arch: dict[str, Any],
    slm_map: dict[int, dict[str, Any]],
) -> list[float]:
    """Compute estimated physical time (μs) for each logical frame.

    - Movement frames: atom_transfer duration per load/store phase.
      Within a phase, time \u221d \u221a(max shuttle distance).
    - CZ frames: rydberg_gate duration.
    - 1Q frames: single_qubit_gate duration \u00d7 max gates on any qubit.
    """
    n_layers = debug["n_layers"]
    placements = debug["placement"]
    routing = debug["routing"]
    sq_layers = debug.get("single_qubit_layers", [])

    op_dur = arch.get("operation_duration", {})
    t_transfer = op_dur.get("atom_transfer", 15.0)
    t_cz = op_dur.get("rydberg_gate", 0.36)
    t_1q = op_dur.get("single_qubit_gate", 0.625)

    # Reference distance for normalisation (diagonal of arch bounding box)
    arch_range = arch.get("arch_range", [[-5, -5], [105, 60]])
    ref_dist = max(1.0, math.hypot(
        arch_range[1][0] - arch_range[0][0],
        arch_range[1][1] - arch_range[0][1],
    ))

    times: list[float] = []
    for f in range(2 * n_layers + 1):
        layer_idx = f // 2
        is_ent = (f % 2 == 1)

        if is_ent:
            times.append(t_cz)
        else:
            t = 0.0
            # 1Q gates
            if layer_idx < len(sq_layers) and sq_layers[layer_idx]:
                gate_counts: dict[int, int] = defaultdict(int)
                for op in sq_layers[layer_idx]:
                    for q in op["qubits"]:
                        gate_counts[q] += 1
                t += t_1q * (max(gate_counts.values()) if gate_counts else 0)

            # STORE movement (from previous entanglement)
            if f > 0 and (f - 1) < len(routing):
                d = _max_shuttle_dist(slm_map, placements[f - 1],
                                      placements[f], routing[f - 1])
                if d > 0:
                    t += t_transfer * math.sqrt(d / ref_dist)

            # LOAD movement (to next entanglement)
            if f < len(routing):
                d = _max_shuttle_dist(slm_map, placements[f],
                                      placements[f + 1], routing[f])
                if d > 0:
                    t += t_transfer * math.sqrt(d / ref_dist)

            times.append(max(t, 0.5))

    return times


def animate_compilation_movie(
    debug: dict[str, Any],
    arch: str | dict[str, Any],
    figsize: tuple[float, float] | None = None,
    repeat: bool = True,
    max_frames: int | None = None,
    time_scale: float = 80.0,
    sub_frames_per_move: int = 10,
    sub_frames_per_gate: int = 6,
    show_circuit: bool = True,
) -> "matplotlib.animation.FuncAnimation":
    """Create a time-proportional *movie* of the entire compilation.

    Unlike :func:`animate_compilation` (which uses a fixed interval),
    this function generates smooth shuttle animations with intermediate
    atom positions and adjusts frame durations so that the playback
    speed is proportional to real physical time.

    Movement phases show atoms gliding from source to destination with
    ease-in-out interpolation.  Gate phases are held for a duration
    proportional to the gate time reported in the architecture spec.

    Args:
        debug: The dictionary returned by ``compiler.debug_info()``.
        arch: Architecture JSON string or dict.
        figsize: Figure size in inches.
        repeat: Whether the animation loops.
        max_frames: Cap the number of logical frames.
        time_scale: Milliseconds of animation per microsecond of physical
                    time.  Increase for slower playback.
        sub_frames_per_move: Number of interpolation sub-frames per
                             shuttle movement phase.
        sub_frames_per_gate: Number of pulse sub-frames used to animate
                             CZ and 1Q phases.
        show_circuit: Show circuit timeline below the main view.

    Returns:
        A :class:`matplotlib.animation.FuncAnimation`.
    """
    try:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as e:
        msg = "matplotlib is required: pip install matplotlib"
        raise ImportError(msg) from e

    arch_data: dict[str, Any] = json.loads(arch) if isinstance(arch, str) else dict(arch)
    slm_map = _build_slm_map(arch_data)
    n_layers = debug["n_layers"]
    n_logical = 2 * n_layers + 1
    if max_frames is not None:
        n_logical = min(n_logical, max_frames)

    routing = debug["routing"]
    placements = debug["placement"]
    sq_layers = debug.get("single_qubit_layers", [])
    two_q_layers = debug.get("two_qubit_layers", [])
    frame_times = _compute_frame_times(debug, arch_data, slm_map)

    # Build expanded frame list.
    # Each entry: (phase, logical_frame, t, group_idx)
    Phase = str  # type alias
    expanded: list[tuple[Phase, int, float, int]] = []

    for f in range(n_logical):
        is_ent = (f % 2 == 1)
        layer_idx = f // 2

        if is_ent:
            for s in range(max(sub_frames_per_gate, 1)):
                t = s / max(sub_frames_per_gate - 1, 1)
                expanded.append(("cz", f, t, 0))
        else:
            has_store = (f > 0 and (f - 1) < len(routing) and
                         any(_qubit_xy(slm_map, placements[f - 1][q]) !=
                             _qubit_xy(slm_map, placements[f][q])
                             for group in routing[f - 1] for q in group))
            has_1q = (layer_idx < len(sq_layers) and
                      len(sq_layers[layer_idx]) > 0)
            has_load = (f < len(routing) and
                        any(_qubit_xy(slm_map, placements[f][q]) !=
                            _qubit_xy(slm_map, placements[f + 1][q])
                            for group in routing[f] for q in group))

            if has_store:
                for g_idx in range(len(routing[f - 1])):
                    for s in range(sub_frames_per_move):
                        t = s / max(sub_frames_per_move - 1, 1)
                        expanded.append(("store_move", f, t, g_idx))

            if has_1q:
                for s in range(max(sub_frames_per_gate, 1)):
                    t = s / max(sub_frames_per_gate - 1, 1)
                    expanded.append(("1q", f, t, 0))

            if has_load:
                for g_idx in range(len(routing[f])):
                    for s in range(sub_frames_per_move):
                        t = s / max(sub_frames_per_move - 1, 1)
                        expanded.append(("load_move", f, t, g_idx))

            # If nothing happened, show one static frame
            if not has_store and not has_1q and not has_load:
                expanded.append(("static", f, 1.0, 0))

    n_expanded = len(expanded)

    # Setup figure
    if show_circuit and n_layers > 0:
        fig = plt.figure(figsize=figsize or (15, 11))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[5.5, 0.8, 1.0],
                      hspace=0.12)
        ax = fig.add_subplot(gs[0])
        ax_timeline = fig.add_subplot(gs[1])
        ax_info = fig.add_subplot(gs[2])
    else:
        fig, ax = plt.subplots(figsize=figsize or (14, 9))
        ax_timeline = None
        ax_info = None

    _xmin, _xmax, _ymin, _ymax = _arch_axis_limits(arch_data)

    last_panel_frame = -1
    anim: Any | None = None

    def _update(exp_idx: int) -> None:
        nonlocal last_panel_frame
        phase, logical_frame, t, group_idx = expanded[exp_idx]

        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("")
        ax.set_ylabel("y (\u00b5m)")
        ax.grid(visible=True, linestyle=":", linewidth=0.4,
                color="#cccccc", zorder=0)

        if phase == "store_move":
            # Interpolate movement from previous entanglement → current storage
            # We render using placement[f-1] → placement[f]
            _render_directional_interpolation(
                ax, arch_data, slm_map, debug, logical_frame,
                direction="store", t=t, group_idx=group_idx)
        elif phase == "load_move":
            # Interpolate movement from current storage → next entanglement
            _render_directional_interpolation(
                ax, arch_data, slm_map, debug, logical_frame,
                direction="load", t=t, group_idx=group_idx)
        elif phase == "cz":
            layer_idx = logical_frame // 2
            frame_pos = {
                q: _qubit_xy(slm_map, placements[logical_frame][q])
                for q in range(debug.get("n_qubits", 0))
            }
            active_qubits = []
            if layer_idx < len(two_q_layers):
                active_qubits = [q for pair in two_q_layers[layer_idx] for q in pair]
            _render_ops_frame(
                ax,
                arch_data,
                slm_map,
                debug,
                {
                    "phase": "cz",
                    "logical_frame": logical_frame,
                    "current_pos": frame_pos,
                    "active_qubits": active_qubits,
                    "active_load": set(),
                    "t": t,
                },
            )
        elif phase == "1q":
            layer_idx = logical_frame // 2
            frame_pos = {
                q: _qubit_xy(slm_map, placements[logical_frame][q])
                for q in range(debug.get("n_qubits", 0))
            }
            active_qubits = []
            if layer_idx < len(sq_layers):
                active_qubits = [q for op in sq_layers[layer_idx] for q in op.get("qubits", [])]
            _render_ops_frame(
                ax,
                arch_data,
                slm_map,
                debug,
                {
                    "phase": "1q",
                    "logical_frame": logical_frame,
                    "current_pos": frame_pos,
                    "active_qubits": active_qubits,
                    "active_load": set(),
                    "t": t,
                    "gate_name": "1Q",
                },
            )
        else:
            _render_frame(ax, arch_data, slm_map, debug, logical_frame)

        ax.set_xlim(_xmin, _xmax)
        ax.set_ylim(_ymax, _ymin)   # fixed limits after render

        is_ent = (logical_frame % 2 == 1) and (logical_frame // 2 < n_layers)
        _build_legends(ax, debug, is_ent)

        if logical_frame != last_panel_frame:
            if ax_timeline is not None:
                ax_timeline.cla()
                _draw_circuit_timeline(ax_timeline, debug, logical_frame)
            if ax_info is not None:
                ax_info.cla()
                _draw_frame_info(ax_info, debug, logical_frame, arch_data, slm_map)
            last_panel_frame = logical_frame

        if anim is not None and (exp_idx + 1) < len(intervals):
            anim.event_source.interval = intervals[exp_idx + 1]

    # Compute per-frame interval from physical times
    intervals: list[float] = []
    for phase, f, t, _group_idx in expanded:
        ft = frame_times[f] if f < len(frame_times) else 1.0
        if phase in ("store_move", "load_move"):
            ms = ft * time_scale / max(sub_frames_per_move, 1)
        elif phase in ("cz", "1q"):
            ms = ft * time_scale / max(sub_frames_per_gate, 1)
        else:
            ms = ft * time_scale
        intervals.append(max(ms, 25))

    first_interval = intervals[0] if intervals else 80

    anim = animation.FuncAnimation(fig, _update, frames=n_expanded,
                                   interval=first_interval, repeat=repeat,
                                   blit=False)
    fig.tight_layout()
    return anim


def _render_directional_interpolation(
    ax: Any,
    arch: dict[str, Any],
    slm_map: dict[int, dict[str, Any]],
    debug: dict[str, Any],
    logical_frame: int,
    direction: str,
    t: float,
    group_idx: int = 0,
) -> None:
    """Render a smooth shuttle movement for the *movie* animation.

    Args:
        direction: ``"store"`` (previous ent. → current storage) or
                   ``"load"`` (current storage → next ent.).
        t: Interpolation parameter in [0, 1].
        group_idx: The routing group currently being interpolated.
    """
    n_qubits: int = debug["n_qubits"]
    placements = debug["placement"]
    routing = debug["routing"]
    n_layers: int = debug["n_layers"]

    # Ease-in-out cubic
    if t < 0.5:
        t_smooth = 4 * t * t * t
    else:
        t_smooth = 1 - (-2 * t + 2) ** 3 / 2

    if direction == "store":
        src_idx = logical_frame - 1
        dst_idx = logical_frame
        route_idx = logical_frame - 1
        label = "\u2193 Storing"
    else:
        src_idx = logical_frame
        dst_idx = logical_frame + 1
        route_idx = logical_frame
        label = "\u2191 Loading"

    src = placements[src_idx] if src_idx >= 0 and src_idx < len(placements) else placements[0]
    dst = placements[dst_idx] if dst_idx < len(placements) else src

    current_group_moving = set()
    moved_qubits = set()
    
    if 0 <= route_idx < len(routing):
        if group_idx < len(routing[route_idx]):
            for q in routing[route_idx][group_idx]:
                if _qubit_xy(slm_map, src[q]) != _qubit_xy(slm_map, dst[q]):
                    current_group_moving.add(q)
        for g in range(group_idx):
            if g < len(routing[route_idx]):
                for q in routing[route_idx][g]:
                    if _qubit_xy(slm_map, src[q]) != _qubit_xy(slm_map, dst[q]):
                        moved_qubits.add(q)

    _draw_background(ax, arch, laser_active=False)

    # Compute current (interpolated) position for every qubit.
    cur_xy: dict[int, tuple[float, float]] = {}
    for q in range(n_qubits):
        x0, y0 = _qubit_xy(slm_map, src[q])
        x1, y1 = _qubit_xy(slm_map, dst[q])
        if q in current_group_moving:
            cur_xy[q] = (x0 + (x1 - x0) * t_smooth, y0 + (y1 - y0) * t_smooth)
        elif q in moved_qubits:
            cur_xy[q] = (x1, y1)
        else:
            cur_xy[q] = (x0, y0)

    # ── Routing-group connectors ───────────────────────────────────────────
    # Qubits in the same routing group are shuttled together as a unit in
    # the AOD (same row or column).  Draw a thin line between them so the
    # compiler's grouping decision is visible during movement.
    if 0 <= route_idx < len(routing) and group_idx < len(routing[route_idx]):
        group = routing[route_idx][group_idx]
        moving_in_group = [q for q in group if q in current_group_moving]
        if len(moving_in_group) >= 2:
            gxs = [cur_xy[q][0] for q in moving_in_group]
            gys = [cur_xy[q][1] for q in moving_in_group]
            ax.plot(gxs, gys, "-", color="#555555",
                    linewidth=0.7, alpha=0.35, zorder=4,
                    solid_capstyle="round")

    for q in current_group_moving:
        x, y = cur_xy[q]
        x0, y0 = _qubit_xy(slm_map, src[q])
        x1, y1 = _qubit_xy(slm_map, dst[q])
        ax.plot([x0, x], [y0, y], "-", color=_qubit_color(q), linewidth=1.2,
                alpha=0.35, zorder=4)

        if t_smooth < 0.85:
            ax.scatter([x0], [y0], s=40, c=[_qubit_color(q)],
                       alpha=0.12, zorder=3, linewidths=0)
        if t_smooth < 0.9:
            ax.scatter([x1], [y1], s=50, facecolors="none",
                       edgecolors=_qubit_color(q), linewidths=0.5,
                       alpha=0.25, zorder=3, linestyles="dotted")

    moving_qubits = [q for q in range(n_qubits) if q in current_group_moving]
    idle_qubits = [q for q in range(n_qubits) if q not in current_group_moving]

    if idle_qubits:
        ax.scatter([cur_xy[q][0] for q in idle_qubits],
                   [cur_xy[q][1] for q in idle_qubits],
                   s=78,
                   c=[_qubit_color(q) for q in idle_qubits],
                   zorder=8,
                   linewidths=0.8,
                   edgecolors="white")
    if moving_qubits:
        ax.scatter([cur_xy[q][0] for q in moving_qubits],
                   [cur_xy[q][1] for q in moving_qubits],
                   s=92,
                   c=[_qubit_color(q) for q in moving_qubits],
                   zorder=8,
                   linewidths=1.2,
                   edgecolors="white")

    if n_qubits <= 64:
        for q in range(n_qubits):
            x, y = cur_xy[q]
            ax.text(x, y, str(q), fontsize=5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=9)

    layer_idx = logical_frame // 2
    moving = set(
        q for g in routing[route_idx] for q in g
        if _qubit_xy(slm_map, src[q]) != _qubit_xy(slm_map, dst[q])
    ) if 0 <= route_idx < len(routing) else set()
    n_moving = len(moving)
    n_groups = sum(
        1 for g in routing[route_idx] if any(q in moving for q in g)
    ) if 0 <= route_idx < len(routing) else 0
    group_note = f", {n_groups} group{'s' if n_groups != 1 else ''}" if n_groups else ""
    ax.set_title(
        f"Shuttling \u2014 {label} \u2014 Layer {layer_idx} \u2014 "
        f"{n_moving} atom{'s' if n_moving != 1 else ''}{group_note} "
        f"({t_smooth * 100:.0f}%)\n"
        f"(frame {logical_frame}/{2 * n_layers})",
        fontsize=9)

def _render_ops_frame(ax, arch_data, slm_map, debug, frame_data):
    import math
    current_pos = frame_data["current_pos"]
    active_qubits = frame_data["active_qubits"]
    active_load = frame_data["active_load"]
    phase = frame_data["phase"]
    t = frame_data.get("t", 1.0)

    _draw_background(ax, arch_data, laser_active=False)

    n_qubits = debug.get("n_qubits", 0)

    # Smooth pulse: 0 at t=0 and t=1, peaks at t=0.5
    pulse = math.sin(math.pi * t)

    if phase == "move":
        start_pos = frame_data["start_pos"]
        end_pos = frame_data["end_pos"]
        for q in active_qubits:
            x0, y0 = start_pos[q]
            x, y = current_pos[q]
            x1, y1 = end_pos[q]
            ax.plot([x0, x], [y0, y], "-", color=_qubit_color(q), linewidth=1.2, alpha=0.35, zorder=4)
            if t < 0.85:
                ax.scatter([x0], [y0], s=40, c=[_qubit_color(q)], alpha=0.12, zorder=3, linewidths=0)
            if t < 0.9:
                ax.scatter([x1], [y1], s=50, facecolors="none", edgecolors=_qubit_color(q),
                           linewidths=0.5, alpha=0.25, zorder=3, linestyles="dotted")

    elif phase == "cz":
        from matplotlib.patches import FancyBboxPatch

        # ── Collect active atoms first ─────────────────────────────────────────
        logical_frame_idx = frame_data["logical_frame"]
        layer = logical_frame_idx // 2
        tq = debug.get("two_qubit_layers", [])
        pairs = tq[layer] if layer < len(tq) else []
        active_in_cz: set[int] = set()
        for pair in pairs:
            active_in_cz.update([int(pair[0]), int(pair[1])])

        # ── Only illuminate EZ zones that contain active atoms ─────────────────
        active_positions = {current_pos[q] for q in active_in_cz if q in current_pos}

        for ez in arch_data.get("entanglement_zones", []):
            off = ez.get("offset", [0, 0])
            dim = ez.get("dimension", [10, 10])
            x0 = off[1] - 0.5
            y0 = off[0] - 0.5
            w = dim[1] + 1
            h = dim[0] + 1

            # Skip zones that don't contain any active atom
            ez_has_active = any(
                x0 <= px <= x0 + w and y0 <= py <= y0 + h
                for px, py in active_positions
            )
            if not ez_has_active:
                continue

            # Main fill — pulsing warm fill within the EZ bounds
            ax.add_patch(FancyBboxPatch(
                (x0, y0), w, h,
                boxstyle="round,pad=0.3",
                facecolor="#ff3300", alpha=0.04 + 0.28 * pulse,
                edgecolor="#ff6600", linewidth=1.0 + 3.5 * pulse,
                zorder=2,
            ))
            # Inner bright core
            ipad = 0.5
            if w > 2 * ipad and h > 2 * ipad:
                ax.add_patch(FancyBboxPatch(
                    (x0 + ipad, y0 + ipad), w - 2 * ipad, h - 2 * ipad,
                    boxstyle="round,pad=0.1",
                    facecolor="#ff8800", alpha=0.0 + 0.15 * pulse,
                    edgecolor="none", zorder=2,
                ))

            # Horizontal scan lines — Rydberg laser field effect
            n_lines = max(2, int(dim[0]))
            for row in range(n_lines + 1):
                y_line = off[0] + row * (dim[0] / max(n_lines, 1))
                row_pulse = pulse * (0.4 + 0.6 * math.sin(math.pi * row / max(n_lines, 1)))
                ax.plot([x0 + 0.5, x0 + w - 0.5], [y_line, y_line],
                        "-", color="#ffcc00",
                        linewidth=0.3 + 0.6 * row_pulse,
                        alpha=0.03 + 0.22 * row_pulse, zorder=3)

        # ── Corona rings: batched scatter (one call per ring layer) ───────────
        _cz_xs = [current_pos[q][0] for q in active_in_cz if q in current_pos]
        _cz_ys = [current_pos[q][1] for q in active_in_cz if q in current_pos]
        if _cz_xs:
            ax.scatter(_cz_xs, _cz_ys, s=900 + 700 * pulse,
                       c="#ff4400", alpha=0.02 + 0.10 * pulse,
                       zorder=5, linewidths=0)
            ax.scatter(_cz_xs, _cz_ys, s=280 + 240 * pulse,
                       facecolors="none", edgecolors="#ff3300",
                       linewidths=1.5 + 2.5 * pulse,
                       alpha=0.15 + 0.75 * pulse, zorder=6)

        active_gate_qubits = active_in_cz

    elif phase == "1q":
        from matplotlib.patches import Arc

        gate_name = frame_data.get("gate_name", "")
        active_gate_qubits: set[int] = set(active_qubits)

        # ── Pulsing halo: one batched scatter call ─────────────────────────────
        _1q_xs = [current_pos[q][0] for q in active_qubits if q in current_pos]
        _1q_ys = [current_pos[q][1] for q in active_qubits if q in current_pos]
        if _1q_xs:
            ax.scatter(_1q_xs, _1q_ys, s=250 + 200 * pulse,
                       facecolors="none", edgecolors="#00cc44",
                       linewidths=1.5 + 2.0 * pulse,
                       alpha=0.20 + 0.70 * pulse, zorder=7)

        # ── Per-atom: rotating arc + label (can't be batched) ─────────────────
        sweep = t * 360.0
        for q in active_qubits:
            if q not in current_pos:
                continue
            x, y = current_pos[q]
            if sweep > 2:
                ax.add_patch(Arc((x, y), width=1.4, height=1.4,
                                 angle=-90, theta1=0, theta2=sweep,
                                 color="#00dd55", linewidth=2.5, alpha=0.85, zorder=8))
            if gate_name:
                ax.text(x, y - 1.7, gate_name.upper(),
                        fontsize=7, color="#00bb44",
                        alpha=0.5 + 0.5 * pulse,
                        fontweight="bold", ha="center", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.35 + 0.45 * pulse,
                                  edgecolor="#00aa33", linewidth=0.8))
    else:
        active_gate_qubits = set()

    # ── Draw all atoms (batched by active/inactive) ────────────────────────────
    if phase == "cz":
        _active_set = active_in_cz
    elif phase == "1q":
        _active_set = set(active_qubits)
    else:
        _active_set = set()

    _idle_xs, _idle_ys, _idle_c = [], [], []
    _act_xs, _act_ys, _act_c = [], [], []
    _load_xs, _load_ys = [], []

    for q in range(n_qubits):
        x, y = current_pos[q]
        if q in active_load:
            _load_xs.append(x); _load_ys.append(y)
        if q in _active_set:
            _act_xs.append(x); _act_ys.append(y); _act_c.append(_qubit_color(q))
        else:
            _idle_xs.append(x); _idle_ys.append(y); _idle_c.append(_qubit_color(q))

    if _load_xs:
        ax.scatter(_load_xs, _load_ys, s=190, facecolors="none",
                   edgecolors="#4b5563", linewidths=0.8,
                   alpha=0.18, zorder=7)
    if _idle_xs:
        ax.scatter(_idle_xs, _idle_ys, s=80, c=_idle_c, zorder=8,
                   linewidths=0.8, edgecolors="white")
    if _act_xs:
        ax.scatter(_act_xs, _act_ys, s=130 + 60 * pulse, c=_act_c, zorder=8,
                   linewidths=2.0 + 1.5 * pulse, edgecolors="white")

    # labels — still per-atom but text is cheap
    for q in range(n_qubits):
        x, y = current_pos[q]
        ax.text(x, y, str(q), fontsize=5 + (1 if q in _active_set else 0),
                ha="center", va="center",
                color="white", fontweight="bold", zorder=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    phase_labels = {"move": f"Shuttle ({t * 100:.0f}%)", "cz": "CZ gate  (Rydberg entanglement)", "1q": "Single-qubit gate"}
    ax.set_title(phase_labels.get(phase, phase.upper()), fontsize=9)
