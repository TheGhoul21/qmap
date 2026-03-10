"""
Pre-warm the presentation_app media cache.

Pre-generates full MP4 movies for selected presets using the exact same
parameters as the Streamlit app, so the first user never waits.

Usage:
    uv run python prewarm_cache.py                    # list available presets
    uv run python prewarm_cache.py --all              # generate all presets
    uv run python prewarm_cache.py "GHZ (6q)" "QFT (8q)"
    uv run python prewarm_cache.py --list             # same as no args

Options:
    --all         Generate all presets (skip Custom QASM)
    --list        List preset names and exit
    --figsize W H Figure size in inches (default: 17 10)
    --sub-frames N Sub-frames per move (default: 4)
    --fps F       Frames per second (default: 10.0)
    --max-frames N Max logical frames (default: all)
    --skip-cached Skip presets that are already cached (default: True)
    --force       Re-generate even if already cached
    --workers N   Parallel workers for frame rendering (default: cpu_count)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

# ── path setup (same as presentation_app.py) ─────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "python"))

import mqt.qmap  # noqa: F401  — anchor namespace before mqt.core
import mqt as _mqt_ns
_local_mqt = str(_HERE / "python" / "mqt")
if _local_mqt not in list(_mqt_ns.__path__):
    _mqt_ns.__path__.insert(0, _local_mqt)

# ── architecture (identical to presentation_app.py) ───────────────────────────
DEFAULT_ARCH_JSON = json.dumps({
    "name": "compact_32q_architecture",
    "operation_duration": {"rydberg_gate": 0.36, "single_qubit_gate": 52, "atom_transfer": 15},
    "operation_fidelity": {"rydberg_gate": 0.995, "single_qubit_gate": 0.9997, "atom_transfer": 0.999},
    "qubit_spec": {"T": 1.5e6},
    "storage_zones": [{"zone_id": 0, "slms": [
        {"id": 0, "site_separation": [6, 6], "r": 4, "c": 8, "location": [0, 0]}
    ], "offset": [0, 0], "dimension": [48, 24]}],
    "entanglement_zones": [{"zone_id": 0, "slms": [
        {"id": 1, "site_separation": [6, 6], "r": 2, "c": 16, "location": [1, 38]},
        {"id": 2, "site_separation": [6, 6], "r": 2, "c": 16, "location": [3, 38]},
    ], "offset": [1, 38], "dimension": [98, 14]}],
    "aods": [{"id": 0, "site_separation": 2, "r": 8, "c": 16}],
    "arch_range": [[-5, -5], [105, 60]],
    "rydberg_range": [[[-5, 33], [105, 60]]],
})

# ── presets (identical to presentation_app.py) ────────────────────────────────
PRESETS: dict[str, Any] = {
    "CNOT chain (5q)": {"builder": "cnot_chain", "n_qubits": 5},
    "Teleportation-like (4q)": {"builder": "teleport_like", "n_qubits": 4},
    "Linear chain (8q)": {"builder": "linear_chain", "n_qubits": 8},
    "GHZ (6q)": {"builder": "ghz", "n_qubits": 6},
    "QFT (8q)": {"builder": "qft", "n_qubits": 8},
    "QFT large (18q)": {"builder": "qft", "n_qubits": 18},
    "QPE phase-Z (10q)": {"builder": "qpe", "n_qubits": 10, "counting_qubits": 9},
    "QPE medium (12q)": {"builder": "qpe", "n_qubits": 12, "counting_qubits": 10},
    "Dense CZ grid (12q)": {"builder": "dense_grid", "n_qubits": 12},
    "Brickwork (10q, 3 layers)": {"builder": "brickwork", "n_qubits": 10, "n_layers": 3},
    "Brickwork large (24q, 4 layers)": {"builder": "brickwork", "n_qubits": 24, "n_layers": 4},
    "Brickwork XL (30q, 5 layers)": {"builder": "brickwork", "n_qubits": 30, "n_layers": 5},
    "QAOA ring (28q, p=6)": {"builder": "qaoa_ring", "n_qubits": 28, "p_layers": 6},
    # "Custom QASM" intentionally omitted — no fixed circuit to pre-generate
}


# ── cache helpers (identical to presentation_app.py) ──────────────────────────
def _cache_dir() -> Path:
    d = _HERE / ".presentation_media_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(kind: str, **parts: Any) -> str:
    payload = {"version": 1, "kind": kind, **parts}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _cache_read(key: str, ext: str) -> bytes | None:
    p = _cache_dir() / f"{key}{ext}"
    return p.read_bytes() if p.exists() else None


def _cache_write(key: str, ext: str, data: bytes) -> None:
    target = _cache_dir() / f"{key}{ext}"
    if target.exists():
        return
    tmp = _cache_dir() / f"{key}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_bytes(data)
        tmp.replace(target)
    except OSError:
        tmp.unlink(missing_ok=True)


# ── circuit builders (identical to presentation_app.py) ───────────────────────
def _build_circuit(preset_name: str, preset: dict[str, Any]):
    import mqt.core.ir as ir
    import math as _math

    def _cp(qc, angle, c, t):
        qc.p(angle / 2, c)
        qc.h(t); qc.cz(c, t); qc.h(t)
        qc.p(-angle / 2, t)
        qc.h(t); qc.cz(c, t); qc.h(t)
        qc.p(angle / 2, t)

    def _swap(qc, a, b):
        qc.h(b); qc.cz(a, b); qc.h(b)
        qc.h(a); qc.cz(b, a); qc.h(a)
        qc.h(b); qc.cz(a, b); qc.h(b)

    b = preset["builder"]

    if b == "linear_chain":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        for i in range(n - 1):
            qc.cz(i, i + 1)
        for i in range(n - 2):
            qc.cz(i, i + 2)
        qc.h(0); qc.h(n - 1)
        return qc

    if b == "ghz":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        qc.h(0)
        for i in range(1, n):
            qc.h(i)
            qc.cz(0, i)
            qc.h(i)
        return qc

    if b == "qft":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                _cp(qc, _math.pi / (2 ** (j - i)), i, j)
        for i in range(n // 2):
            _swap(qc, i, n - 1 - i)
        return qc

    if b == "qpe":
        n = preset["n_qubits"]
        n_count = preset.get("counting_qubits", n - 1)
        qc = ir.QuantumComputation(n)
        target = n_count
        phi = _math.pi / 4  # T gate phase
        qc.x(target)
        for q in range(n_count):
            qc.h(q)
        for k in range(n_count):
            _cp(qc, phi * (1 << k), k, target)
        for i in range(n_count // 2):
            _swap(qc, i, n_count - 1 - i)
        for i in range(n_count - 1, -1, -1):
            qc.h(i)
            for j in range(i - 1, -1, -1):
                _cp(qc, -_math.pi / (2 ** (i - j)), j, i)
        return qc

    if b == "qaoa_ring":
        n = preset["n_qubits"]
        p = preset.get("p_layers", 4)
        qc = ir.QuantumComputation(n)
        for _ in range(p):
            for q in range(n):
                qc.h(q)
            for i in range(0, n - 1, 2):
                qc.cz(i, i + 1)
            for i in range(1, n - 1, 2):
                qc.cz(i, i + 1)
            qc.cz(n - 1, 0)
            for q in range(n):
                qc.h(q)
        return qc

    if b == "dense_grid":
        n = preset["n_qubits"]
        side = int(n**0.5)
        qc = ir.QuantumComputation(n)
        for i in range(n - 1):
            qc.cz(i, i + 1)
        for i in range(n - side):
            qc.cz(i, i + side)
        for i in range(n):
            qc.h(i)
        return qc

    if b == "brickwork":
        n = preset["n_qubits"]
        nl = preset.get("n_layers", 3)
        qc = ir.QuantumComputation(n)
        for _ in range(nl):
            for i in range(0, n - 1, 2):
                qc.cz(i, i + 1)
            for i in range(1, n - 1, 2):
                qc.cz(i, i + 1)
            for i in range(n):
                qc.h(i)
        return qc

    if b == "cnot_chain":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        qc.h(0)
        for i in range(n - 1):
            qc.h(i + 1); qc.cz(i, i + 1); qc.h(i + 1)
        qc.h(n - 1)
        return qc

    if b == "teleport_like":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        qc.h(0)
        qc.h(1); qc.cz(0, 1); qc.h(1)
        qc.h(2); qc.cz(1, 2); qc.h(2)
        qc.h(2)
        qc.h(3); qc.cz(2, 3); qc.h(3)
        qc.h(1)
        qc.h(3); qc.cz(0, 3); qc.h(3)
        return qc

    raise ValueError(f"Unknown builder: {b}")


# ── pre-generation ────────────────────────────────────────────────────────────
def prewarm(
    preset_name: str,
    figsize_w: float = 17.0,
    figsize_h: float = 10.0,
    sub_frames: int = 4,
    fps: float = 10.0,
    max_frames: int | None = None,
    force: bool = False,
    n_workers: int = 1,
    dpi: int = 80,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import mqt.qmap.na.zoned as zoned
    from mqt.qmap.visualization import save_compilation_animation

    preset = PRESETS[preset_name]
    arch = zoned.ZonedNeutralAtomArchitecture.from_json_string(DEFAULT_ARCH_JSON)

    print(f"  Compiling '{preset_name}'...", flush=True)
    qc = _build_circuit(preset_name, preset)
    c = zoned.RoutingAwareCompiler(arch)
    c.compile(qc)
    debug = c.debug_info()
    debug_json = json.dumps(debug)  # must match: json.dumps(di) in presentation_app.py line 1007

    key = _cache_key(
        "full_mp4",
        debug=debug_json,
        arch=DEFAULT_ARCH_JSON,
        figsize_w=figsize_w,
        figsize_h=figsize_h,
        sub_frames=sub_frames,
        fps=fps,
        max_frames=max_frames,
        gate_duration_s=0.9,
    )

    cached_path = _cache_dir() / f"{key}.mp4"
    if cached_path.exists() and not force:
        size_mb = cached_path.stat().st_size / 1024 / 1024
        print(f"  Already cached ({size_mb:.1f} MB) — skipping.", flush=True)
        return

    print(f"  Rendering MP4 (n_layers={debug['n_layers']})...", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        save_compilation_animation(
            debug, DEFAULT_ARCH_JSON, tmp_path,
            fps=fps,
            figsize=(figsize_w, figsize_h),
            dpi=dpi,
            max_frames=max_frames,
            start_frame=0,
            show_circuit=True,
            smooth_movement=True,
            sub_frames_per_move=sub_frames,
            sub_frames_per_gate=max(10, sub_frames * 3),
            gate_duration_s=0.9,
            verbose=True,
            n_workers=n_workers,
        )
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    _cache_write(key, ".mp4", data)
    size_mb = len(data) / 1024 / 1024
    print(f"  Saved {size_mb:.1f} MB → {cached_path.name}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-warm presentation_app full MP4 cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "presets", nargs="*",
        help="Preset names to generate (use quotes for names with spaces).",
    )
    parser.add_argument("--all", action="store_true", help="Generate all presets.")
    parser.add_argument("--list", action="store_true", help="List available presets and exit.")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"), default=[17.0, 10.0])
    parser.add_argument("--sub-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max logical frames (0 or omit = all).")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if already cached.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for frame rendering (default: cpu_count).")
    parser.add_argument("--dpi", type=int, default=80,
                        help="Raster DPI for frames (default: 80). Not in cache key — "
                             "lower = faster render, slightly lower resolution.")

    args = parser.parse_args()

    if args.list or (not args.presets and not args.all):
        print("Available presets:")
        for name in PRESETS:
            print(f"  {name!r}")
        return

    import os as _os
    max_frames = args.max_frames if (args.max_frames and args.max_frames > 0) else None
    n_workers = args.workers if args.workers is not None else _os.cpu_count() or 1

    if args.all:
        targets = list(PRESETS.keys())
    else:
        targets = args.presets
        unknown = [p for p in targets if p not in PRESETS]
        if unknown:
            for u in unknown:
                print(f"Unknown preset: {u!r}", file=sys.stderr)
            print("Run without arguments to see available presets.", file=sys.stderr)
            sys.exit(1)

    print(f"Cache dir: {_cache_dir()}")
    print(f"Settings: figsize={args.figsize}, dpi={args.dpi}, sub_frames={args.sub_frames}, "
          f"fps={args.fps}, max_frames={max_frames}, force={args.force}, workers={n_workers}")
    print()

    for i, name in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {name}")
        try:
            prewarm(
                name,
                figsize_w=args.figsize[0],
                figsize_h=args.figsize[1],
                sub_frames=args.sub_frames,
                fps=args.fps,
                max_frames=max_frames,
                force=args.force,
                n_workers=n_workers,
                dpi=args.dpi,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
