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
    # 1Q gates: glow ring on atom + compact label floating just above.
    # No tall beams — everything stays within the zone bounding box.
    sq_layers: list[list[dict[str, Any]]] = debug.get("single_qubit_layers", [])
    sq_on_qubit: dict[int, list[tuple[str, list[float]]]] = {}
    if not is_entanglement_frame and layer_idx < len(sq_layers):
        for op in sq_layers[layer_idx]:
            entry = (op["name"], op.get("params", []))
            for q in op["qubits"]:
                sq_on_qubit.setdefault(q, []).append(entry)

        # For large circuits (many qubits with 1Q gates), avoid label clutter.
        # If more than MAX_LABELED qubits get labels, fall back to glow-only.
        MAX_LABELED = 12
        n_affected = len(sq_on_qubit)
        show_labels = n_affected <= MAX_LABELED

        for q_idx, (q, ops) in enumerate(sorted(sq_on_qubit.items())):
            x, y = _qubit_xy(slm_map, current_placement[q])

            # Outer glow ring (dashed yellow circle around atom)
            ax.scatter([x], [y], s=420, facecolors="#f1c40f", alpha=0.15,
                       zorder=7, linewidths=0)
            ax.scatter([x], [y], s=260, facecolors="none",
                       edgecolors="#f1c40f", linewidths=1.5,
                       linestyles="dashed", zorder=10, alpha=0.9)

            if show_labels:
                # Compact label: gate name floats just above the atom.
                # Alternate slightly left/right to reduce column overlap.
                parts = [_format_gate(name, params) for name, params in ops]
                label = ", ".join(parts)
                x_shift = 2.5 if (q_idx % 2 == 0) else -2.5
                ax.annotate(
                    label,
                    xy=(x, y), xytext=(x + x_shift, y - 3.5),
                    fontsize=4.5, ha="center", va="bottom",
                    color="#7d6608", fontweight="bold", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.18", fc="#fffde7",
                              ec="#f1c40f", alpha=0.92, linewidth=0.8),
                    arrowprops=dict(arrowstyle="-", color="#f1c40f",
                                    lw=0.5, alpha=0.5),
                )

        if not show_labels:
            # Summarise: group identical gate signatures and show a compact note.
            gate_groups: dict[str, list[int]] = {}
            for q, ops in sq_on_qubit.items():
                sig = ", ".join(_format_gate(n, p) for n, p in ops)
                gate_groups.setdefault(sig, []).append(q)
            summary_parts = [
                f"{sig} \u00d7{len(qs)}" for sig, qs in gate_groups.items()
            ]
            summary = "1Q gates: " + ";  ".join(summary_parts)
            ax.text(0.5, 0.02, summary, fontsize=5.5, ha="center", va="bottom",
                    transform=ax.transAxes, color="#7d6608", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fffde7",
                              ec="#f1c40f", alpha=0.92, linewidth=0.8))

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
    """Draw a qubit × layer grid showing the compiled gate schedule.

    Rows = qubits.  Column pairs = layers (1Q phase | 2Q phase).
    - Yellow cell: qubit has a 1Q gate in that layer.
    - Red cell:    qubit is in a CZ pair for that layer.
    - Green border: qubit is reused (stays in ent. zone).
    - Current layer pair is highlighted with a thick red border.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    n_layers = debug["n_layers"]
    n_qubits = debug["n_qubits"]
    tq_layers = debug["two_qubit_layers"]
    sq_layers = debug.get("single_qubit_layers", [])
    reuse = debug.get("reuse_qubits", [])

    cur_layer = frame_idx // 2

    # Build lookup: layer → set of qubits with 1Q gates
    sq_qubits: list[set[int]] = []
    for i in range(n_layers):
        s: set[int] = set()
        if i < len(sq_layers):
            for op in sq_layers[i]:
                s.update(op["qubits"])
        sq_qubits.append(s)

    # Build lookup: layer → set of qubits in CZ pairs
    tq_qubits: list[set[int]] = []
    for i in range(n_layers):
        s = set()
        for pair in tq_layers[i]:
            s.update(pair)
        tq_qubits.append(s)

    MAX_VIS_LAYERS = 24
    MAX_LABELED_QUBITS = 16   # show qubit labels on Y axis up to this count
    # Compact mode: cells too small to fit text when many qubits or many layers
    compact = n_qubits > MAX_LABELED_QUBITS

    if n_layers > MAX_VIS_LAYERS:
        half = MAX_VIS_LAYERS // 2
        l_start = max(0, min(cur_layer - half, n_layers - MAX_VIS_LAYERS))
        l_end = l_start + MAX_VIS_LAYERS
    else:
        l_start, l_end = 0, n_layers

    vis_layers = range(l_start, l_end)
    n_vis = len(vis_layers)

    # Cell size: shrink when many qubits to keep the grid compact.
    cell_w = 1.0
    cell_h = min(1.0, 24.0 / max(n_qubits, 1))
    total_w = n_vis * 2 * cell_w
    total_h = n_qubits * cell_h

    ax.set_xlim(-1.5, total_w + 1.5)
    ax.set_ylim(-0.3, total_h + 0.3)
    ax.invert_yaxis()
    if not compact:
        ax.set_yticks([q * cell_h + cell_h / 2 for q in range(n_qubits)])
        ax.set_yticklabels([f"q{q}" for q in range(n_qubits)], fontsize=4)
    else:
        # For many qubits, label only every Nth qubit
        step = max(1, n_qubits // 8)
        ax.set_yticks([q * cell_h + cell_h / 2 for q in range(0, n_qubits, step)])
        ax.set_yticklabels([f"q{q}" for q in range(0, n_qubits, step)], fontsize=4)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Layer labels along top (only every other when compact)
    label_step = 2 if n_vis > 12 else 1
    for vi, i in enumerate(vis_layers):
        if vi % label_step == 0:
            col_x = vi * 2 * cell_w
            ax.text(col_x + cell_w, -0.2, f"L{i}",
                    fontsize=3.5, ha="center", va="bottom",
                    color="#444444", fontweight="bold")

    # Draw cells
    reuse_set_per_layer: list[set[int]] = [
        set(reuse[i]) if i < len(reuse) else set() for i in range(n_layers)
    ]

    for vi, i in enumerate(vis_layers):
        col_1q = vi * 2 * cell_w
        col_2q = col_1q + cell_w
        is_cur = (i == cur_layer)

        for q in range(n_qubits):
            row_y = q * cell_h

            # 1Q cell
            has_1q = q in sq_qubits[i]
            fc_1q = "#f9e154" if has_1q else "#f5f5f5"
            ec_1q = "#b7950b" if has_1q else "#e0e0e0"
            ax.add_patch(Rectangle(
                (col_1q, row_y), cell_w, cell_h,
                facecolor=fc_1q, edgecolor=ec_1q, linewidth=0.4, zorder=2))
            if has_1q and not compact and cell_h >= 0.6:
                gates = [_format_gate(op["name"], op.get("params", []))
                         for op in sq_layers[i] if q in op["qubits"]]
                ax.text(col_1q + cell_w / 2, row_y + cell_h / 2,
                        ",".join(gates),
                        fontsize=3, ha="center", va="center",
                        color="#7d6608", fontweight="bold", zorder=3)

            # 2Q cell
            has_2q = q in tq_qubits[i]
            is_reused = q in reuse_set_per_layer[i]
            fc_2q = "#f1948a" if has_2q else "#f5f5f5"
            lw_2q = 1.2 if is_reused else 0.4
            ec_2q = "#2ecc71" if is_reused else ("#c0392b" if has_2q else "#e0e0e0")
            ax.add_patch(Rectangle(
                (col_2q, row_y), cell_w, cell_h,
                facecolor=fc_2q, edgecolor=ec_2q, linewidth=lw_2q, zorder=2))
            if has_2q and not compact and cell_h >= 0.6:
                ax.text(col_2q + cell_w / 2, row_y + cell_h / 2, "CZ",
                        fontsize=3, ha="center", va="center",
                        color="#922b21", fontweight="bold", zorder=3)

        # Highlight current active phase with a thick border
        if is_cur:
            hi_col = col_1q if (frame_idx % 2 == 0) else col_2q
            ax.add_patch(Rectangle(
                (hi_col, 0), cell_w, total_h,
                facecolor="none", edgecolor="#e74c3c", linewidth=2,
                linestyle="-", zorder=5))

    # Compact inline legend to the right
    ax.text(total_w + 0.2, total_h * 0.15, "\u25a0 1Q",
            fontsize=3.5, va="center", ha="left", color="#b7950b")
    ax.text(total_w + 0.2, total_h * 0.50, "\u25a0 CZ",
            fontsize=3.5, va="center", ha="left", color="#c0392b")
    ax.text(total_w + 0.2, total_h * 0.85, "\u25a0 reuse",
            fontsize=3.5, va="center", ha="left", color="#2ecc71")

    suffix = f"  (L{l_start}\u2013L{l_end - 1} of {n_layers - 1})" \
        if n_layers > MAX_VIS_LAYERS else ""
    ax.set_title(f"Compiled gate schedule{suffix}", fontsize=6, pad=2)


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
        fig = plt.figure(figsize=figsize or (19.2, 10.8), constrained_layout=True)
        gs = GridSpec(
            3, 2, figure=fig,
            height_ratios=[5.5, 1.6, 0.7],
            width_ratios=[4.5, 1],
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_overview = fig.add_subplot(gs[1, 0])
        ax_timeline = fig.add_subplot(gs[2, 0])
        ax_right = fig.add_subplot(gs[:, 1])   # full right column: legends + info
    else:
        fig, ax = plt.subplots(figsize=figsize or (14, 9), constrained_layout=True)
        ax_overview = None
        ax_timeline = None
        ax_right = None

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("y (\u00b5m)")
    ax.grid(visible=True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    _render_frame(ax, arch_data, slm_map, debug, frame)
    ax.autoscale_view()

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


def animate_compilation(
    debug: dict[str, Any],
    arch: str | dict[str, Any],
    interval: int = 800,
    figsize: tuple[float, float] | None = None,
    repeat: bool = True,
    max_frames: int | None = None,
    show_circuit: bool = True,
) -> "matplotlib.animation.FuncAnimation":
    """Create an animated step-through of the full compilation.

    Each frame alternates between storage placements (with movement arrows
    showing both LOAD and STORE shuttle paths) and entanglement-zone
    placements (with CZ highlights). Total frames = ``2 * n_layers + 1``.

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
        interval: Milliseconds between frames.
        figsize: Figure size in inches.
        repeat: Whether the animation loops.
        max_frames: Cap the number of frames rendered (useful for large
                    circuits). ``None`` renders all frames.
        show_circuit: If True, show circuit timeline and info panels.

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
    n_frames = 2 * n_layers + 1
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    if show_circuit and n_layers > 0:
        fig = plt.figure(figsize=figsize or (19.2, 10.8), constrained_layout=True)
        gs = GridSpec(
            3, 2, figure=fig,
            height_ratios=[5.5, 1.6, 0.7],
            width_ratios=[4.5, 1],
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_overview = fig.add_subplot(gs[1, 0])
        ax_timeline = fig.add_subplot(gs[2, 0])
        ax_right = fig.add_subplot(gs[:, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize or (14, 9), constrained_layout=True)
        ax_overview = None
        ax_timeline = None
        ax_right = None

    def _update(frame: int) -> None:
        ax.cla()
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("")
        ax.set_ylabel("y (\u00b5m)")
        ax.grid(visible=True, linestyle=":", linewidth=0.4,
                color="#cccccc", zorder=0)
        _render_frame(ax, arch_data, slm_map, debug, frame)
        ax.autoscale_view()

        is_ent = (frame % 2 == 1) and (frame // 2 < n_layers)
        if ax_right is not None:
            ax_right.cla()
            _build_legends(ax_right, debug, is_ent)
            _draw_frame_info(ax_right, debug, frame, arch_data, slm_map,
                             y_top=0.48)

        if ax_overview is not None:
            ax_overview.cla()
            _draw_circuit_overview(ax_overview, debug, frame)
        if ax_timeline is not None:
            ax_timeline.cla()
            _draw_circuit_timeline(ax_timeline, debug, frame)

    anim = animation.FuncAnimation(fig, _update, frames=n_frames,
                                   interval=interval, repeat=repeat,
                                   blit=False)
    return anim


def save_compilation_animation(
    debug: dict[str, Any],
    arch: str | dict[str, Any],
    path: str,
    fps: float = 2.0,
    figsize: tuple[float, float] | None = None,
    dpi: int = 80,
    max_frames: int | None = None,
    show_circuit: bool = True,
    verbose: bool = True,
) -> None:
    """Save a compilation animation directly to a GIF or MP4 file.

    Unlike :func:`animate_compilation` (which uses FuncAnimation and keeps
    the entire animation in memory), this function renders each frame
    independently with :func:`visualize_compilation_step`, closes the
    matplotlib figure immediately, and writes only the compressed pixel data.
    Memory usage stays roughly constant regardless of frame count.

    For GIF output (path ending in ``.gif``) ``pillow`` is required.
    For MP4 output (path ending in ``.mp4``) ``ffmpeg`` is required.

    Recommended parameters for large circuits:
      - ``figsize=(12, 6.75)`` — half-resolution (1280×720 at 96 dpi)
      - ``dpi=80`` — reduces per-frame pixel count vs the 1920×1080 default
      - ``max_frames=50`` — cap long animations during exploration

    Args:
        debug: The dictionary returned by ``compiler.debug_info()``.
        arch: Architecture JSON string or dict.
        path: Output file path. Extension determines format (``.gif``/``.mp4``).
        fps: Frames per second.
        figsize: Figure size in inches. Defaults to ``(12, 6.75)`` (≈1280×720).
        dpi: Dots per inch when rasterising each frame.
        max_frames: If set, stop after this many frames.
        show_circuit: Passed through to :func:`visualize_compilation_step`.
        verbose: Print progress every 10 frames.
    """
    try:
        import io
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        msg = "pillow and matplotlib are required: pip install pillow matplotlib"
        raise ImportError(msg) from e

    arch_data: dict[str, Any] = json.loads(arch) if isinstance(arch, str) else dict(arch)
    n_layers = debug["n_layers"]
    n_frames = 2 * n_layers + 1
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    _anim_figsize = figsize or (12.0, 6.75)   # 1280×~540 at 80dpi — much smaller than 1920×1080
    path_lower = path.lower()
    use_ffmpeg = path_lower.endswith(".mp4") or path_lower.endswith(".webm")

    if use_ffmpeg:
        import subprocess, tempfile, os
        tmpdir = tempfile.mkdtemp(prefix="qmap_anim_")
        try:
            for f in range(n_frames):
                fig = visualize_compilation_step(
                    debug, arch_data, frame=f,
                    figsize=_anim_figsize, show_circuit=show_circuit,
                )
                fig.savefig(f"{tmpdir}/frame_{f:05d}.png", dpi=dpi)
                plt.close(fig)
                if verbose and (f + 1) % 10 == 0:
                    print(f"  rendered {f + 1}/{n_frames} frames...")
            cmd = [
                "ffmpeg", "-y", "-r", str(fps),
                "-i", f"{tmpdir}/frame_%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
            ]
            subprocess.run(cmd, check=True, capture_output=not verbose)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        # GIF: render each frame → PIL image → close figure → accumulate PIL images.
        # PIL images are ~10-50× smaller in memory than the matplotlib figure object.
        pil_frames: list[Image.Image] = []
        for f in range(n_frames):
            fig = visualize_compilation_step(
                debug, arch_data, frame=f,
                figsize=_anim_figsize, show_circuit=show_circuit,
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            plt.close(fig)          # <── release figure memory immediately
            buf.seek(0)
            pil_frames.append(Image.open(buf).copy())
            buf.close()
            if verbose and (f + 1) % 10 == 0:
                print(f"  rendered {f + 1}/{n_frames} frames...")

        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )

    if verbose:
        import os
        size_mb = os.path.getsize(path) / 1e6
        print(f"Saved {path}  ({n_frames} frames @ {fps} fps, {size_mb:.1f} MB)")


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
    frame_times = _compute_frame_times(debug, arch_data, slm_map)

    # Build expanded frame list.
    # Each entry: (phase, logical_frame, t)
    #   phase: "store_move", "1q", "load_move", "cz", "static"
    #   t: interpolation parameter [0, 1] for movement sub-frames
    Phase = str  # type alias
    expanded: list[tuple[Phase, int, float]] = []

    for f in range(n_logical):
        is_ent = (f % 2 == 1)
        layer_idx = f // 2

        if is_ent:
            expanded.append(("cz", f, 1.0))
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
                for s in range(sub_frames_per_move):
                    t = s / max(sub_frames_per_move - 1, 1)
                    expanded.append(("store_move", f, t))

            if has_1q:
                expanded.append(("1q", f, 1.0))

            if has_load:
                for s in range(sub_frames_per_move):
                    t = s / max(sub_frames_per_move - 1, 1)
                    expanded.append(("load_move", f, t))

            # If nothing happened, show one static frame
            if not has_store and not has_1q and not has_load:
                expanded.append(("static", f, 1.0))

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

    def _update(exp_idx: int) -> None:
        phase, logical_frame, t = expanded[exp_idx]

        ax.cla()
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("")
        ax.set_ylabel("y (\u00b5m)")
        ax.grid(visible=True, linestyle=":", linewidth=0.4,
                color="#cccccc", zorder=0)

        if phase == "store_move":
            # Interpolate movement from previous entanglement → current storage
            # We render using placement[f-1] → placement[f]
            _render_directional_interpolation(
                ax, arch_data, slm_map, debug, logical_frame,
                direction="store", t=t)
        elif phase == "load_move":
            # Interpolate movement from current storage → next entanglement
            _render_directional_interpolation(
                ax, arch_data, slm_map, debug, logical_frame,
                direction="load", t=t)
        elif phase == "cz":
            _render_frame(ax, arch_data, slm_map, debug, logical_frame)
        elif phase == "1q":
            _render_frame(ax, arch_data, slm_map, debug, logical_frame)
        else:
            _render_frame(ax, arch_data, slm_map, debug, logical_frame)

        is_ent = (logical_frame % 2 == 1) and (logical_frame // 2 < n_layers)
        _build_legends(ax, debug, is_ent)
        ax.autoscale_view()

        if ax_timeline is not None:
            ax_timeline.cla()
            _draw_circuit_timeline(ax_timeline, debug, logical_frame)
        if ax_info is not None:
            ax_info.cla()
            _draw_frame_info(ax_info, debug, logical_frame, arch_data, slm_map)

    # Compute per-frame interval from physical times
    intervals: list[float] = []
    for phase, f, t in expanded:
        ft = frame_times[f] if f < len(frame_times) else 1.0
        if phase in ("store_move", "load_move"):
            ms = ft * time_scale / max(sub_frames_per_move, 1)
        else:
            ms = ft * time_scale
        intervals.append(max(ms, 25))

    avg_interval = sum(intervals) / max(len(intervals), 1)

    anim = animation.FuncAnimation(fig, _update, frames=n_expanded,
                                   interval=avg_interval, repeat=repeat,
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
) -> None:
    """Render a smooth shuttle movement for the *movie* animation.

    Args:
        direction: ``"store"`` (previous ent. → current storage) or
                   ``"load"`` (current storage → next ent.).
        t: Interpolation parameter in [0, 1].
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

    moving: set[int] = set()
    if 0 <= route_idx < len(routing):
        for group in routing[route_idx]:
            for q in group:
                if _qubit_xy(slm_map, src[q]) != _qubit_xy(slm_map, dst[q]):
                    moving.add(q)

    _draw_background(ax, arch, laser_active=False)

    for q in range(n_qubits):
        x0, y0 = _qubit_xy(slm_map, src[q])
        if q in moving:
            x1, y1 = _qubit_xy(slm_map, dst[q])
            x = x0 + (x1 - x0) * t_smooth
            y = y0 + (y1 - y0) * t_smooth
            # Trail
            ax.plot([x0, x], [y0, y], "-", color=_qubit_color(q),
                    linewidth=1.2, alpha=0.25, zorder=4)
            # Ghost at source
            if t_smooth < 0.85:
                ax.scatter([x0], [y0], s=40, c=[_qubit_color(q)],
                           alpha=0.12, zorder=3, linewidths=0)
            # Target marker
            if t_smooth < 0.9:
                ax.scatter([x1], [y1], s=50, facecolors="none",
                           edgecolors=_qubit_color(q), linewidths=0.5,
                           alpha=0.25, zorder=3, linestyles="dotted")
        else:
            # Non-moving atoms: show at their position in the destination frame
            # (they stay put, but their "current" placement depends on direction)
            if direction == "store":
                x, y = _qubit_xy(slm_map, dst[q])
            else:
                x, y = x0, y0

        ax.scatter([x], [y], s=80, c=[_qubit_color(q)], zorder=8,
                   linewidths=0.8, edgecolors="white")
        ax.text(x, y, str(q), fontsize=5, ha="center", va="center",
                color="white", fontweight="bold", zorder=9)

    layer_idx = logical_frame // 2
    n_moving = len(moving)
    ax.set_title(
        f"Shuttling \u2014 {label} \u2014 Layer {layer_idx} \u2014 "
        f"{n_moving} atom{'s' if n_moving != 1 else ''} "
        f"({t_smooth * 100:.0f}%)\n"
        f"(frame {logical_frame}/{2 * n_layers})",
        fontsize=9)
