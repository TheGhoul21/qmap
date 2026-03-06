"""
Quantum Compilation Presentation App
=====================================
Streamlit interactive explorer for zoned neutral-atom compilation.

Features:
  - Preset circuits + free-form QASM input
  - Real-time compilation with progress feedback
  - Layer-by-layer navigation (prev / next / slider)
  - Per-frame detail panel (gates, timing, CZ pairs)
  - Auto-play slideshow mode for live presentation

Run:
    uv run streamlit run presentation_app.py
"""

from __future__ import annotations

import io
import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "python"))

# Import mqt.qmap first to anchor the namespace before any mqt.core import.
# Python namespace packages (PEP 420) are order-sensitive: whichever subpackage
# is imported first "wins" and fixes the mqt.__path__.  If mqt.core arrives
# first the local python/ tree is excluded and mqt.qmap becomes unfindable.
import mqt.qmap  # noqa: F401

# Also repair __path__ in case mqt was already imported by a previous run in
# the same process (e.g. after Streamlit hot-reload).
import mqt as _mqt_ns
_local_mqt = str(Path(__file__).parent / "python" / "mqt")
if _local_mqt not in list(_mqt_ns.__path__):
    _mqt_ns.__path__.insert(0, _local_mqt)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Compilation Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── default architecture ───────────────────────────────────────────────────────
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

# ── preset circuits ────────────────────────────────────────────────────────────
PRESETS: dict[str, Any] = {
    "CNOT chain (5q)": {
        "description": "Catena di CNOT — mostra la decomposizione c-U → CZ",
        "builder": "cnot_chain",
        "n_qubits": 5,
        "display_qasm": (
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[5];\n"
            "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncx q[3],q[4];\nh q[4];\n"
        ),
    },
    "Teleportation-like (4q)": {
        "description": "Circuito con CNOT e porte miste → CZ basis",
        "builder": "teleport_like",
        "n_qubits": 4,
        "display_qasm": (
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n"
            "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\nh q[2];\ncx q[2],q[3];\nh q[1];\ncx q[0],q[3];\n"
        ),
    },
    "Linear chain (8q)": {
        "description": "8-qubit linear CZ chain",
        "builder": "linear_chain",
        "n_qubits": 8,
    },
    "GHZ (6q)": {
        "description": "GHZ state preparation",
        "builder": "ghz",
        "n_qubits": 6,
    },
    "QFT (8q)": {
        "description": "Quantum Fourier Transform su 8 qubit",
        "builder": "qft",
        "n_qubits": 8,
    },
    "Dense CZ grid (12q)": {
        "description": "Griglia densa di CZ — stress test del router",
        "builder": "dense_grid",
        "n_qubits": 12,
    },
    "Brickwork (10q, 3 layers)": {
        "description": "Pattern a brickwork — circuiti variazionali",
        "builder": "brickwork",
        "n_qubits": 10,
        "n_layers": 3,
    },
    "Brickwork large (24q, 4 layers)": {
        "description": "Preset piu corposo ma ancora gestibile su laptop",
        "builder": "brickwork",
        "n_qubits": 24,
        "n_layers": 4,
    },
    "Custom QASM": {
        "description": "Scrivi il tuo circuito in OpenQASM 3",
        "builder": "qasm",
    },
}


def _build_circuit(preset_name: str, preset: dict[str, Any], qasm_text: str):
    """Return a QuantumComputation for the given preset/QASM."""
    import mqt.core.ir as ir

    b = preset["builder"]
    if b == "linear_chain":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        for i in range(n - 1):
            qc.cz(i, i + 1)
        for i in range(n - 2):
            qc.cz(i, i + 2)
        qc.h(0)
        qc.h(n - 1)
        return qc, n

    if b == "ghz":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cz(0, i + 1)
        for i in range(n):
            qc.h(i)
        return qc, n

    if b == "qft":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                qc.cz(i, j)
        return qc, n

    if b == "dense_grid":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        side = int(n**0.5)
        for i in range(n - 1):
            qc.cz(i, i + 1)
        for i in range(n - side):
            qc.cz(i, i + side)
        for i in range(n):
            qc.h(i)
        return qc, n

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
        return qc, n

    if b == "cnot_chain":
        # Compile with CZ (CNOT = H·CZ·H on target); display keeps CNOT QASM
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        qc.h(0)
        for i in range(n - 1):
            qc.h(i + 1)
            qc.cz(i, i + 1)
            qc.h(i + 1)
        qc.h(n - 1)
        return qc, n

    if b == "teleport_like":
        n = preset["n_qubits"]
        qc = ir.QuantumComputation(n)
        qc.h(0)
        # cx q[0],q[1]
        qc.h(1); qc.cz(0, 1); qc.h(1)
        # cx q[1],q[2]
        qc.h(2); qc.cz(1, 2); qc.h(2)
        qc.h(2)
        # cx q[2],q[3]
        qc.h(3); qc.cz(2, 3); qc.h(3)
        qc.h(1)
        # cx q[0],q[3]
        qc.h(3); qc.cz(0, 3); qc.h(3)
        return qc, n

    if b == "qasm":
        qc = ir.QuantumComputation.from_qasm_str(qasm_text)
        return qc, qc.num_qubits

    raise ValueError(f"Unknown builder: {b}")


# ── compilation ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compile_circuit(preset_name: str, qasm_text: str, arch_json: str) -> dict[str, Any]:
    """Compile and return debug_info dict. Cached by inputs."""
    import mqt.qmap.na.zoned as zoned

    arch = zoned.ZonedNeutralAtomArchitecture.from_json_string(arch_json)
    preset = PRESETS[preset_name]
    qc, _ = _build_circuit(preset_name, preset, qasm_text)
    c = zoned.RoutingAwareCompiler(arch)
    c.compile(qc)
    return c.debug_info()


# ── quantikz circuit rendering ────────────────────────────────────────────────
_GATE_MAP = {
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "t": "T", "sdg": r"S^\dagger", "tdg": r"T^\dagger",
    "rx": r"R_x", "ry": r"R_y", "rz": r"R_z",
    "u": "U", "u1": r"U_1", "u2": r"U_2", "u3": r"U_3",
}


def _build_quantikz_source(debug: dict) -> str:
    """Generate a standalone LaTeX document with the compiled circuit."""
    n = debug["n_qubits"]
    n_layers = debug["n_layers"]
    tq_layers = debug.get("two_qubit_layers", [])
    sq_layers = debug.get("single_qubit_layers", [])

    columns: list[dict[int, str]] = []

    for layer in range(n_layers):
        # 1Q gate column (skip if empty)
        sq = sq_layers[layer] if layer < len(sq_layers) else []
        if sq:
            col: dict[int, str] = {}
            for op in sq:
                name = _GATE_MAP.get(op["name"].lower(), op["name"].upper())
                for q in op.get("qubits", []):
                    col[q] = rf"\gate{{{name}}}"
            columns.append(col)

        # CZ column (skip if empty)
        tq = tq_layers[layer] if layer < len(tq_layers) else []
        if tq:
            col = {}
            for pair in tq:
                i, j = int(pair[0]), int(pair[1])
                if i > j:
                    i, j = j, i
                col[i] = rf"\ctrl{{{j - i}}}"
                col[j] = rf"\ctrl{{{i - j}}}"
            columns.append(col)

    rows = []
    for q in range(n):
        cells = [rf"\lstick{{$q_{{{q}}}$}}"]
        for col in columns:
            cells.append(col.get(q, r"\qw"))
        cells.append(r"\qw")
        rows.append(" & ".join(cells))

    body = " \\\\\n".join(rows)
    # scale down for wide circuits
    scale = max(0.5, min(1.0, 12.0 / max(1, len(columns))))
    return (
        r"\documentclass[border=4pt]{standalone}" + "\n"
        r"\usepackage{quantikz}" + "\n"
        r"\begin{document}" + "\n"
        rf"\scalebox{{{scale:.2f}}}{{" + "\n"
        r"\begin{quantikz}[column sep=0.35cm, row sep=0.45cm]" + "\n"
        f"{body}\n"
        r"\end{quantikz}" + "\n"
        "}\n"
        r"\end{document}"
    )


def _parse_qasm_to_columns(qasm_str: str, n_qubits: int) -> list[dict[int, str]]:
    """Parse a QASM 2.0 string into quantikz column dicts."""
    import re
    columns: list[dict[int, str]] = []
    skip = ("OPENQASM", "include", "qreg", "creg", "//", "barrier", "measure")
    for raw in qasm_str.splitlines():
        line = raw.strip().rstrip(";").strip()
        if not line or any(line.startswith(p) for p in skip):
            continue
        # Two-qubit: name[(params)] q[i], q[j]
        m2 = re.match(
            r'([a-z][a-z0-9_]*)(?:\([^)]*\))?\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]$',
            line, re.IGNORECASE,
        )
        if m2:
            name, qi, qj = m2.group(1).lower(), int(m2.group(2)), int(m2.group(3))
            d = qj - qi
            col: dict[int, str] = {}
            if name in ("cx", "cnot"):
                col[qi] = rf"\ctrl{{{d}}}"
                col[qj] = r"\targ{}"
            elif name == "cz":
                col[qi] = rf"\ctrl{{{d}}}"
                col[qj] = rf"\ctrl{{{-d}}}"
            elif name.startswith("c") and len(name) > 1:
                inner = _GATE_MAP.get(name[1:], name[1:].upper())
                col[qi] = rf"\ctrl{{{d}}}"
                col[qj] = rf"\gate{{{inner}}}"
            else:
                label = _GATE_MAP.get(name, name.upper())
                col[qi] = rf"\gate{{{label}}}"
                col[qj] = rf"\gate{{{label}}}"
            columns.append(col)
            continue
        # Single-qubit: name[(params)] q[i]
        m1 = re.match(
            r'([a-z][a-z0-9_]*)(?:\([^)]*\))?\s+q\[(\d+)\]$',
            line, re.IGNORECASE,
        )
        if m1:
            name, qi = m1.group(1).lower(), int(m1.group(2))
            label = _GATE_MAP.get(name, name.upper())
            columns.append({qi: rf"\gate{{{label}}}"})
    return columns


@st.cache_data(show_spinner=False)
def render_original_circuit_png(original_qasm: str, n_qubits: int) -> bytes | None:
    """Render the original (pre-compilation) circuit as PNG via quantikz.

    Takes the QASM string directly (extracted at compile time) so that no
    ``mqt`` imports are needed here, avoiding namespace-package conflicts.
    """
    import os
    import subprocess
    import tempfile

    if not original_qasm.strip():
        return None

    columns = _parse_qasm_to_columns(original_qasm, n_qubits)
    n = n_qubits
    if not columns:
        return None

    scale = max(0.4, min(1.0, 12.0 / max(1, len(columns))))
    rows = []
    for q in range(n):
        cells = [rf"\lstick{{$q_{{{q}}}$}}"]
        for col in columns:
            cells.append(col.get(q, r"\qw"))
        cells.append(r"\qw")
        rows.append(" & ".join(cells))
    body = " \\\\\n".join(rows)
    latex_src = (
        r"\documentclass[border=4pt]{standalone}" + "\n"
        r"\usepackage{quantikz}" + "\n"
        r"\begin{document}" + "\n"
        rf"\scalebox{{{scale:.2f}}}{{" + "\n"
        r"\begin{quantikz}[column sep=0.35cm, row sep=0.45cm]" + "\n"
        f"{body}\n"
        r"\end{quantikz}" + "\n"
        "}\n"
        r"\end{document}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex = os.path.join(tmpdir, "orig.tex")
        pdf = os.path.join(tmpdir, "orig.pdf")
        png = os.path.join(tmpdir, "orig.png")
        with open(tex, "w") as f:
            f.write(latex_src)
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "orig.tex"],
            cwd=tmpdir, capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        subprocess.run(
            ["magick", "-density", "220", pdf,
             "-background", "white", "-alpha", "remove", "-alpha", "off", png],
            check=True, capture_output=True, timeout=30,
        )
        with open(png, "rb") as f:
            return f.read()


@st.cache_data(show_spinner=False)
def render_circuit_png(debug_json: str) -> bytes:
    """Compile the quantikz circuit to PNG via pdflatex + magick."""
    import os
    import subprocess
    import tempfile

    debug = json.loads(debug_json)
    latex_src = _build_quantikz_source(debug)

    with tempfile.TemporaryDirectory() as tmpdir:
        tex = os.path.join(tmpdir, "circuit.tex")
        pdf = os.path.join(tmpdir, "circuit.pdf")
        png = os.path.join(tmpdir, "circuit.png")

        with open(tex, "w") as f:
            f.write(latex_src)

        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "circuit.tex"],
            cwd=tmpdir, capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            raise RuntimeError(f"pdflatex failed:\n{r.stdout[-800:]}")

        subprocess.run(
            ["magick", "-density", "220", pdf,
             "-background", "white", "-alpha", "remove", "-alpha", "off", png],
            check=True, capture_output=True, timeout=30,
        )

        with open(png, "rb") as f:
            return f.read()


# ── static frame rendering ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=128)
def render_frame_png(
    debug_json: str, arch_json: str, frame: int,
    figsize_w: float, figsize_h: float,
) -> bytes:
    """Render a single frame to PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mqt.qmap.visualization import visualize_compilation_step

    debug = json.loads(debug_json)
    fig = visualize_compilation_step(
        debug, arch_json,
        frame=frame,
        figsize=(figsize_w, figsize_h),
        show_circuit=True,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── micro-movie helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=32)
def render_layer_gif(
    debug_json: str,
    arch_json: str,
    frame: int,
    figsize_w: float,
    figsize_h: float,
    sub_frames: int = 4,
    fps: float = 8.0,
) -> bytes:
    """Render a smooth micro-movie GIF for a single compilation step.

    Uses the full debug dict so the circuit panel shows the complete circuit
    with the current step highlighted. Only one logical frame is animated
    (start_frame / max_frames).
    """
    import os
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    from mqt.qmap.visualization import save_compilation_animation

    debug = json.loads(debug_json)
    n_layers = debug["n_layers"]
    total_frames = 2 * n_layers + 1
    start = max(0, min(frame, total_frames - 1))
    end = min(start + 1, total_frames)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        tmp_path = f.name
    try:
        save_compilation_animation(
            debug, arch_json, tmp_path,
            fps=fps,
            figsize=(figsize_w, figsize_h),
            dpi=96,
            max_frames=end,
            start_frame=start,
            show_circuit=True,
            smooth_movement=True,
            sub_frames_per_move=sub_frames,
            sub_frames_per_gate=max(10, sub_frames * 3),
            gate_duration_s=0.9,
            verbose=False,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


@st.cache_data(show_spinner=False, max_entries=32)
def render_layer_mp4(
    debug_json: str,
    arch_json: str,
    frame: int,
    figsize_w: float,
    figsize_h: float,
    sub_frames: int = 4,
    fps: float = 10.0,
) -> bytes:
    """Render a smooth micro-movie MP4 for a single compilation step."""
    import os
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    from mqt.qmap.visualization import save_compilation_animation

    debug = json.loads(debug_json)
    n_layers = debug["n_layers"]
    total_frames = 2 * n_layers + 1
    start = max(0, min(frame, total_frames - 1))
    end = min(start + 1, total_frames)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        save_compilation_animation(
            debug, arch_json, tmp_path,
            fps=fps,
            figsize=(figsize_w, figsize_h),
            dpi=96,
            max_frames=end,
            start_frame=start,
            show_circuit=True,
            smooth_movement=True,
            sub_frames_per_move=sub_frames,
            sub_frames_per_gate=max(10, sub_frames * 3),
            gate_duration_s=0.9,
            verbose=False,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


@st.cache_data(show_spinner=False, max_entries=8)
def render_full_mp4(
    debug_json: str,
    arch_json: str,
    figsize_w: float,
    figsize_h: float,
    sub_frames: int = 4,
    fps: float = 10.0,
    max_frames: int | None = None,
) -> bytes:
    """Render a full-compilation MP4 movie (all logical frames by default)."""
    import os
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    from mqt.qmap.visualization import save_compilation_animation

    debug = json.loads(debug_json)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        save_compilation_animation(
            debug,
            arch_json,
            tmp_path,
            fps=fps,
            figsize=(figsize_w, figsize_h),
            dpi=96,
            max_frames=max_frames,
            start_frame=0,
            show_circuit=True,
            smooth_movement=True,
            sub_frames_per_move=sub_frames,
            sub_frames_per_gate=max(10, sub_frames * 3),
            gate_duration_s=0.9,
            verbose=False,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def render_minimal_video_player(
    mp4_bytes: bytes,
    figsize_h: float,
    autoplay: bool = True,
    loop: bool = True,
    muted: bool = True,
) -> None:
    """Render a minimal HTML5 video player with custom controls.

    Controls include only play/pause, progress bar, and fullscreen.
    """
    import base64

    b64 = base64.b64encode(mp4_bytes).decode()
    video_id = f"qv_{uuid.uuid4().hex}"
    play_label = "Pause" if autoplay else "Play"

    attrs: list[str] = ["playsinline", "preload=\"auto\""]
    if autoplay:
        attrs.append("autoplay")
    if loop:
        attrs.append("loop")
    if muted:
        attrs.append("muted")

    html = f"""
<div style="width:100%;">
  <video id="{video_id}" {' '.join(attrs)}
         style="width:100%;border-radius:4px;display:block;background:transparent;box-shadow:none;outline:none"
         src="data:video/mp4;base64,{b64}"></video>
  <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
    <button id="{video_id}_pp"
            style="padding:4px 10px;border:1px solid #d1d5db;border-radius:6px;background:#fff;cursor:pointer;">
      {play_label}
    </button>
    <input id="{video_id}_progress" type="range" min="0" max="1000" value="0"
           style="flex:1;accent-color:#ef4444;">
    <button id="{video_id}_fs"
            style="padding:4px 10px;border:1px solid #d1d5db;border-radius:6px;background:#fff;cursor:pointer;">
      Fullscreen
    </button>
  </div>
</div>
<script>
  (() => {{
    const v = document.getElementById('{video_id}');
    const pp = document.getElementById('{video_id}_pp');
    const pr = document.getElementById('{video_id}_progress');
    const fs = document.getElementById('{video_id}_fs');

    const updatePP = () => {{ pp.textContent = v.paused ? 'Play' : 'Pause'; }};

    pp.addEventListener('click', () => {{
      if (v.paused) v.play(); else v.pause();
      updatePP();
    }});

    v.addEventListener('play', updatePP);
    v.addEventListener('pause', updatePP);

    v.addEventListener('timeupdate', () => {{
      if (!v.duration || !isFinite(v.duration)) return;
      pr.value = Math.floor((v.currentTime / v.duration) * 1000);
    }});

    pr.addEventListener('input', () => {{
      if (!v.duration || !isFinite(v.duration)) return;
      v.currentTime = (Number(pr.value) / 1000) * v.duration;
    }});

    fs.addEventListener('click', async () => {{
      try {{
        if (document.fullscreenElement) {{
          await document.exitFullscreen();
        }} else if (v.requestFullscreen) {{
          await v.requestFullscreen();
        }}
      }} catch (e) {{}}
    }});

    updatePP();
  }})();
</script>
"""

    component_height = int(max(240, figsize_h * 90 + 70))
    components.html(html, height=component_height, scrolling=False)


# ── background preloading ─────────────────────────────────────────────────────

def _preload_worker(
    debug_json: str,
    arch_json: str,
    n_layers: int,
    fig_w: float,
    fig_h: float,
    sub_frames: int,
    gif_fps: float,
    status: dict,          # shared mutable dict: layer -> True|False
    cancel: threading.Event,
) -> None:
    """Generate all layer GIFs sequentially in a background thread."""
    for layer in range(n_layers):
        if cancel.is_set():
            break
        try:
            render_layer_gif(debug_json, arch_json, 2 * layer, fig_w, fig_h, sub_frames, gif_fps)
            status[layer] = True
        except Exception:
            status[layer] = False   # mark done even on failure so button re-enables


def _start_preload(
    debug_json: str,
    arch_json: str,
    n_layers: int,
    fig_w: float,
    fig_h: float,
    sub_frames: int,
    gif_fps: float,
) -> None:
    """Cancel any running preload and start a fresh one."""
    # Cancel previous thread if running
    old_cancel: threading.Event | None = st.session_state.get("preload_cancel")
    if old_cancel is not None:
        old_cancel.set()

    status: dict = {}
    cancel = threading.Event()
    st.session_state["gif_status"] = status
    st.session_state["preload_cancel"] = cancel
    st.session_state["_gif_status_snap"] = {}

    t = threading.Thread(
        target=_preload_worker,
        args=(debug_json, arch_json, n_layers, fig_w, fig_h, sub_frames, gif_fps, status, cancel),
        daemon=True,
    )
    t.start()


# ── misc helpers ───────────────────────────────────────────────────────────────
def _frame_label(frame: int, n_layers: int) -> str:
    if frame == 2 * n_layers:
        return "Posizionamento finale"
    layer = frame // 2
    if frame % 2 == 0:
        return f"Layer {layer} — Shuttle + gate 1Q"
    return f"Layer {layer} — Gate CZ (Rydberg)"


def _build_frame_table(debug: dict) -> list[dict]:
    n_layers = debug["n_layers"]
    tq = debug["two_qubit_layers"]
    sq = debug.get("single_qubit_layers", [])
    rows = []
    for f in range(2 * n_layers + 1):
        layer = f // 2
        if f == 2 * n_layers:
            rows.append({"Frame": f, "Type": "Final", "Layer": layer, "Gates": "-"})
        elif f % 2 == 1:
            n_cz = len(tq[layer]) if layer < len(tq) else 0
            rows.append({"Frame": f, "Type": "CZ", "Layer": layer, "Gates": f"{n_cz} CZ"})
        else:
            n_1q = len(sq[layer]) if layer < len(sq) else 0
            tag = f"{n_1q} 1Q" if n_1q else "-"
            rows.append({"Frame": f, "Type": "Transit", "Layer": layer, "Gates": tag})
    return rows


def _goto(frame: int) -> None:
    st.session_state["frame"] = frame
    st.session_state["_frame_slider"] = frame


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.session_state.setdefault("frame", 0)
    st.session_state.setdefault("_frame_slider", 0)
    st.session_state.setdefault("compiled", False)
    st.session_state.setdefault("autoplay", False)
    st.session_state.setdefault("autoplay_interval", 1.5)
    st.session_state.setdefault("_view_mode", "Micro-movie (MP4)")
    st.session_state.setdefault("_sub_frames", 4)
    st.session_state.setdefault("_gif_fps", 10.0)
    st.session_state.setdefault("_video_autoplay", False)
    st.session_state.setdefault("_video_loop", False)
    st.session_state.setdefault("_video_muted", True)
    st.session_state.setdefault("_video_controls", False)
    st.session_state.setdefault("_full_movie_max_frames", 0)

    # ─────────────────────────────── SIDEBAR ──────────────────────────────────
    with st.sidebar:
        st.title("Quantum Compiler")

        # ── Circuit picker ────────────────────────────────────────────────────
        preset_name = st.selectbox("Circuito", list(PRESETS.keys()))
        preset = PRESETS[preset_name]
        st.caption(f"_{preset['description']}_")

        qasm_text = ""
        if preset["builder"] == "qasm":
            qasm_text = st.text_area(
                "OpenQASM 3 source",
                value=("OPENQASM 3.0;\nqubit[6] q;\nh q[0];\n"
                       "cz q[0], q[1];\ncz q[1], q[2];\ncz q[2], q[3];\n"
                       "cz q[3], q[4];\ncz q[4], q[5];\nh q[2];\nh q[4];\n"),
                height=180,
            )

        if st.button("Compila", type="primary", use_container_width=True):
            st.session_state["compiled"] = False
            _goto(0)
            st.session_state["autoplay"] = False
            with st.spinner("Compilazione in corso..."):
                try:
                    di = compile_circuit(preset_name, qasm_text, DEFAULT_ARCH_JSON)
                    st.session_state["debug_info"] = di
                    st.session_state["debug_json"] = json.dumps(di)
                    st.session_state["compiled"] = True
                    st.session_state["n_layers"] = di["n_layers"]
                    st.session_state["total_frames"] = 2 * di["n_layers"] + 1
                    st.success(f"Compilato: {di['n_layers']} layer, {di['n_qubits']} qubit")
                    # Store original (pre-decomposition) QASM for display
                    _display_qasm = PRESETS[preset_name].get("display_qasm", "")
                    if not _display_qasm:
                        # For presets without explicit display_qasm, extract from compiled qc
                        try:
                            _qc, _ = _build_circuit(preset_name, PRESETS[preset_name], qasm_text)
                            for _m in ("qasm2_str", "qasm_str"):
                                try:
                                    _display_qasm = getattr(_qc, _m)()
                                    if _display_qasm:
                                        break
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    st.session_state["original_qasm"] = _display_qasm
                    st.session_state["original_n_qubits"] = di["n_qubits"]
                except Exception as exc:
                    st.error(f"Errore:\n```\n{exc}\n```")

        # ── Navigation + controls (solo dopo compilazione) ────────────────────
        if st.session_state["compiled"]:
            n_layers: int = st.session_state["n_layers"]
            total: int = st.session_state["total_frames"]
            cur: int = st.session_state["frame"]

            with st.expander("Navigazione", expanded=True):
                slider_val: int = st.slider(
                    "Step (Shuttle ↔ CZ)", 0, total - 1, cur, key="_frame_slider",
                )
                if slider_val != cur:
                    st.session_state["frame"] = slider_val
                    st.rerun()

                st.caption("Navigazione semplice: uno step alla volta, oppure salta di un layer.")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.button("Step -", use_container_width=True,
                              disabled=(cur == 0), on_click=_goto, args=(cur - 1,))
                with c2:
                    st.button("Step +", use_container_width=True,
                              disabled=(cur >= total - 1), on_click=_goto, args=(cur + 1,))
                with c3:
                    st.button("Start", use_container_width=True,
                              on_click=_goto, args=(0,))

                c4, c5 = st.columns(2)
                with c4:
                    st.button("Layer -", use_container_width=True,
                              disabled=(cur < 2), on_click=_goto, args=(cur - 2,))
                with c5:
                    st.button("Layer +", use_container_width=True,
                              disabled=(cur >= total - 2), on_click=_goto, args=(cur + 2,))

            with st.expander("Auto-play"):
                view_mode_side = st.session_state.get("_view_mode", "Micro-movie (MP4)")
                full_movie_mode = view_mode_side == "Movie completa (MP4)"
                st.session_state["autoplay_interval"] = st.slider(
                    "Intervallo (sec)", 0.5, 5.0,
                    float(st.session_state["autoplay_interval"]), 0.1,
                    key="_interval_slider",
                )
                ap1, ap2 = st.columns(2)
                with ap1:
                    if st.button("Play step-by-step", use_container_width=True,
                                 disabled=st.session_state["autoplay"] or full_movie_mode):
                        st.session_state["autoplay"] = True
                        st.rerun()
                with ap2:
                    if st.button("Stop", use_container_width=True,
                                 disabled=not st.session_state["autoplay"]):
                        st.session_state["autoplay"] = False
                        st.rerun()
                if full_movie_mode:
                    st.caption("In Movie completa usa i controlli del player video.")

            with st.expander("Opzioni"):
                st.slider("Larghezza figura", 10.0, 24.0, 17.0, 0.5, key="_fig_w")
                st.slider("Altezza figura", 7.0, 16.0, 10.0, 0.5, key="_fig_h")
                st.radio(
                    "Modalita",
                    ["Micro-movie (MP4)", "Movie completa (MP4)", "Micro-movie (GIF)", "Frame statico"],
                    key="_view_mode",
                )
                if st.session_state.get("_view_mode") in (
                    "Micro-movie (GIF)",
                    "Micro-movie (MP4)",
                    "Movie completa (MP4)",
                ):
                    st.slider("Sub-frame", 2, 12, 4, 1, key="_sub_frames")
                    st.slider("FPS", 4.0, 20.0, 10.0, 1.0, key="_gif_fps")
                if st.session_state.get("_view_mode") == "Movie completa (MP4)":
                    st.slider(
                        "Max frame logici (0 = tutti)",
                        0,
                        400,
                        int(st.session_state.get("_full_movie_max_frames", 0)),
                        5,
                        key="_full_movie_max_frames",
                    )
                if st.session_state.get("_view_mode") in ("Micro-movie (MP4)", "Movie completa (MP4)"):
                    st.toggle("Video autoplay", key="_video_autoplay")
                    st.toggle("Video loop", key="_video_loop")
                    st.toggle("Video muted", key="_video_muted")
                    st.toggle("Controlli video (overlay)", key="_video_controls")
                    st.caption("Overlay OFF: player minimale (play/pause, progress, fullscreen)")

    # ─────────────────────────────── MAIN AREA ────────────────────────────────
    if not st.session_state["compiled"]:
        st.title("Quantum Compilation Explorer")
        st.markdown(
            """
            Benvenuto! Questa app ti permette di **esplorare layer per layer**
            la compilazione di circuiti quantistici su architetture a **atomi neutri zonati**.

            ### Come usarla
            1. **Scegli** un circuito dal menu a sinistra (o scrivi il tuo in QASM)
            2. Premi **Compila**
            3. Naviga i frame con lo slider, i bottoni Prev/Next, o i jump L0, L1, ...
            4. Usa **Auto-play** per una presentazione fluida

            ### Struttura dei frame
            | Frame pari | Frame dispari |
            |:----------:|:-------------:|
            | Atomi in storage + frecce shuttle + gate 1Q | Atomi in entanglement zone + gate CZ |

            > Ogni coppia (frame pari + dispari) = un **layer** della compilazione.
            """
        )
        st.info("Seleziona un circuito e premi Compila per iniziare.")
        return

    # ── Shared state ───────────────────────────────────────────────────────────
    frame: int = st.session_state["frame"]
    n_layers: int = st.session_state["n_layers"]
    total: int = st.session_state["total_frames"]
    di: dict = st.session_state["debug_info"]
    fig_w: float = st.session_state.get("_fig_w", 17.0)
    fig_h: float = st.session_state.get("_fig_h", 10.0)
    autoplay: bool = st.session_state["autoplay"]
    interval: float = st.session_state["autoplay_interval"]
    dbg_json: str = st.session_state["debug_json"]

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.title(preset_name)
        st.markdown(f"**{_frame_label(frame, n_layers)}**")
    with col_badge:
        st.metric("Fase", "CZ Rydberg" if frame % 2 == 1 else "Shuttle")
        st.metric("Layer", f"{frame // 2} / {n_layers - 1}")

    # ── Circuito completo (quantikz) ───────────────────────────────────────────
    with st.expander("Circuito completo", expanded=True):
        col_orig, col_mid, col_comp = st.columns([10, 1, 10])

        with col_orig:
            st.markdown("**Circuito logico** *(input)*")
            with st.spinner(""):
                try:
                    _orig_qasm = st.session_state.get("original_qasm", "")
                    _orig_n = st.session_state.get("original_n_qubits", di["n_qubits"])
                    orig_png = render_original_circuit_png(_orig_qasm, _orig_n)
                    if orig_png:
                        st.image(orig_png, use_container_width=True)
                    else:
                        st.caption("—")
                except Exception:
                    st.caption("—")

        with col_mid:
            st.markdown(
                "<div style='height:100%;display:flex;flex-direction:column;"
                "align-items:center;justify-content:center;padding-top:3rem'>"
                "<div style='font-size:2rem;color:#888'>→</div>"
                "<div style='font-size:0.65rem;color:#aaa;text-align:center;"
                "margin-top:0.3rem'>Rydberg<br>compiler</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_comp:
            st.markdown("**Circuito compilato** *(CZ basis)*")
            with st.spinner(""):
                try:
                    circ_png = render_circuit_png(dbg_json)
                    st.image(circ_png, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Rendering non riuscito: {exc}")

        # ── Compilation stats ────────────────────────────────────────────────
        n_cz = sum(len(pairs) for pairs in di.get("two_qubit_layers", []))
        n_1q = sum(len(ops) for ops in di.get("single_qubit_layers", []))
        st.markdown(
            f"<div style='text-align:center;color:#666;font-size:0.8rem;margin-top:0.5rem'>"
            f"Compilato in <b>{n_cz} CZ</b> (gate Rydberg)"
            f" + <b>{n_1q}</b> gate a singolo qubit"
            f" su <b>{di['n_layers']}</b> layer"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Visualization fragment ─────────────────────────────────────────────────
    view_mode_now = st.session_state.get("_view_mode", "Micro-movie (MP4)")
    run_every: float | None = interval if (autoplay and view_mode_now != "Movie completa (MP4)") else None

    @st.fragment(run_every=run_every)
    def _viz() -> None:
        view_mode = st.session_state.get("_view_mode", "Micro-movie (MP4)")

        if st.session_state["autoplay"] and view_mode != "Movie completa (MP4)":
            cur = st.session_state["frame"]
            tot = st.session_state["total_frames"]
            if cur < tot - 1:
                st.session_state["frame"] = cur + 1
            else:
                st.session_state["autoplay"] = False
                st.rerun()

        f = st.session_state["frame"]
        nl = st.session_state["n_layers"]
        tot = st.session_state["total_frames"]
        fw = st.session_state.get("_fig_w", 17.0)
        fh = st.session_state.get("_fig_h", 10.0)
        dbg_json = st.session_state["debug_json"]
        debug = st.session_state["debug_info"]
        layer = f // 2

        # ── Render ────────────────────────────────────────────────────────────
        if view_mode == "Micro-movie (MP4)":
            sub_frames = st.session_state.get("_sub_frames", 4)
            gif_fps = st.session_state.get("_gif_fps", 10.0)
            with st.spinner(f"Rendering micro-movie step {f}..."):
                try:
                    mp4 = render_layer_mp4(
                        dbg_json, DEFAULT_ARCH_JSON,
                        f, fw, fh,
                        sub_frames=sub_frames,
                        fps=gif_fps,
                    )
                    autoplay = st.session_state.get("_video_autoplay", True)
                    loop = st.session_state.get("_video_loop", True)
                    muted = st.session_state.get("_video_muted", True)
                    controls = st.session_state.get("_video_controls", False)

                    if controls:
                        st.video(
                            mp4,
                            format="video/mp4",
                            autoplay=autoplay,
                            loop=loop,
                            muted=muted,
                        )
                    else:
                        render_minimal_video_player(
                            mp4,
                            figsize_h=fh,
                            autoplay=autoplay,
                            loop=loop,
                            muted=muted,
                        )
                except Exception as exc:
                    st.error(f"Errore micro-movie: {exc}")
                    return
        elif view_mode == "Movie completa (MP4)":
            sub_frames = st.session_state.get("_sub_frames", 4)
            gif_fps = st.session_state.get("_gif_fps", 10.0)
            max_frames_ui = int(st.session_state.get("_full_movie_max_frames", 0))
            max_frames = None if max_frames_ui <= 0 else max_frames_ui
            with st.spinner("Rendering movie completa..."):
                try:
                    mp4 = render_full_mp4(
                        dbg_json,
                        DEFAULT_ARCH_JSON,
                        fw,
                        fh,
                        sub_frames=sub_frames,
                        fps=gif_fps,
                        max_frames=max_frames,
                    )
                    autoplay = st.session_state.get("_video_autoplay", False)
                    loop = st.session_state.get("_video_loop", False)
                    muted = st.session_state.get("_video_muted", True)
                    controls = st.session_state.get("_video_controls", False)

                    if controls:
                        st.video(
                            mp4,
                            format="video/mp4",
                            autoplay=autoplay,
                            loop=loop,
                            muted=muted,
                        )
                    else:
                        render_minimal_video_player(
                            mp4,
                            figsize_h=fh,
                            autoplay=autoplay,
                            loop=loop,
                            muted=muted,
                        )
                except Exception as exc:
                    st.error(f"Errore movie completa: {exc}")
                    return
        elif view_mode == "Micro-movie (GIF)":
            sub_frames = st.session_state.get("_sub_frames", 4)
            gif_fps = st.session_state.get("_gif_fps", 10.0)
            with st.spinner(f"Rendering micro-movie step {f}..."):
                try:
                    gif = render_layer_gif(
                        dbg_json, DEFAULT_ARCH_JSON,
                        f, fw, fh,
                        sub_frames=sub_frames,
                        fps=gif_fps,
                    )
                    import base64
                    b64 = base64.b64encode(gif).decode()
                    st.markdown(
                        f'<img src="data:image/gif;base64,{b64}" '
                        f'style="width:100%;border-radius:4px">',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(f"Errore micro-movie: {exc}")
                    return
        else:
            with st.spinner(f"Rendering frame {f}..."):
                try:
                    png = render_frame_png(dbg_json, DEFAULT_ARCH_JSON, f, fw, fh)
                    st.image(png, use_container_width=True)
                except Exception as exc:
                    st.error(f"Errore rendering: {exc}")
                    return

        # ── Quick-nav (compact) ───────────────────────────────────────────────
        st.markdown("---")
        if view_mode != "Movie completa (MP4)":
            st.caption("Flusso consigliato: Step + per alternare Shuttle → CZ → Shuttle...")
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.button("◀ Step", key="qnav_prev_step", use_container_width=True,
                          disabled=(f == 0), on_click=_goto, args=(f - 1,))
            with q2:
                st.button("Step ▶", key="qnav_next_step", use_container_width=True,
                          disabled=(f >= tot - 1), on_click=_goto, args=(f + 1,))
            with q3:
                st.button("◀ Layer", key="qnav_prev_layer", use_container_width=True,
                          disabled=(f < 2), on_click=_goto, args=(f - 2,))
            with q4:
                st.button("Layer ▶", key="qnav_next_layer", use_container_width=True,
                          disabled=(f >= tot - 2), on_click=_goto, args=(f + 2,))
        else:
            st.caption("Movie completa: usa Play/Pause + barra progress per seguire step-by-step.")

        # ── Detail panel ──────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("Dettagli frame corrente", expanded=True):
            import math, collections
            is_ent = f % 2 == 1

            dc1, dc2, dc3 = st.columns(3)

            with dc1:
                st.markdown("**Info**")
                st.write(f"- Qubit: **{debug['n_qubits']}**")
                st.write(f"- Layer totali: **{nl}**")
                st.write(f"- Frame: **{f}** / {tot - 1}")
                st.write(f"- Layer: **{layer}**")
                st.write(f"- Tipo: **{'CZ (Rydberg)' if is_ent else 'Transit / 1Q'}**")

            with dc2:
                if is_ent:
                    tq = debug.get("two_qubit_layers", [])
                    pairs = tq[layer] if layer < len(tq) else []
                    st.markdown(f"**CZ — layer {layer}** ({len(pairs)} coppie)")
                    for p in pairs:
                        st.write(f"  CZ(q{p[0]}, q{p[1]})")
                    reuse = debug.get("reuse_qubits", [])
                    if layer < len(reuse) and reuse[layer]:
                        rq = ", ".join(f"q{q}" for q in reuse[layer])
                        st.info(f"Riusati: {rq}")
                else:
                    sq = debug.get("single_qubit_layers", [])
                    ops = sq[layer] if layer < len(sq) else []
                    if ops:
                        st.markdown(f"**Gate 1Q — layer {layer}** ({len(ops)})")
                        for op in ops:
                            params = op.get("params", [])
                            def _fa(p: float) -> str:
                                pi = math.pi
                                for d in [1, 2, 4]:
                                    if abs(p * d / pi - round(p * d / pi)) < 1e-6:
                                        n = round(p * d / pi)
                                        return (f"{n}/{d}pi" if d > 1
                                                else (f"{n}pi" if n != 1 else "pi"))
                                return f"{p:.3f}"
                            gate = op["name"]
                            if params:
                                gate += f"({', '.join(_fa(p) for p in params)})"
                            qbs = ", ".join(f"q{q}" for q in op["qubits"])
                            st.write(f"  {gate} -> {qbs}")
                    else:
                        st.info("Nessun gate 1Q in questo layer.")

            with dc3:
                st.markdown("**Shuttle**")
                routing = debug.get("routing", [])
                if not is_ent:
                    if f > 0 and (f - 1) < len(routing):
                        n_s = sum(len(g) for g in routing[f - 1])
                        if n_s:
                            st.write(f"Stored: **{n_s}** atomi")
                    if f < len(routing):
                        n_l = sum(len(g) for g in routing[f])
                        if n_l:
                            st.write(f"Loading: **{n_l}** atomi")
                    if (f >= len(routing) or
                            sum(len(g) for g in routing[f]) == 0) and f == 0:
                        st.info("Frame iniziale.")

                arch_data = json.loads(DEFAULT_ARCH_JSON)
                op_dur = arch_data.get("operation_duration", {})
                t_cz = op_dur.get("rydberg_gate", 0.36)
                t_1q = op_dur.get("single_qubit_gate", 52.0)
                t_tr = op_dur.get("atom_transfer", 15.0)
                if is_ent:
                    st.metric("Tempo stimato", f"{t_cz:.2f} us")
                else:
                    sq = debug.get("single_qubit_layers", [])
                    ops = sq[layer] if layer < len(sq) else []
                    qc: dict[int, int] = collections.defaultdict(int)
                    for op in ops:
                        for q in op["qubits"]:
                            qc[q] += 1
                    t_est = t_1q * (max(qc.values()) if qc else 0)
                    if f < len(routing) and sum(len(g) for g in routing[f]) > 0:
                        t_est += t_tr
                    if (f > 0 and (f - 1) < len(routing)
                            and sum(len(g) for g in routing[f - 1]) > 0):
                        t_est += t_tr
                    if t_est > 0:
                        st.metric("Tempo stimato", f"{t_est:.1f} us")

        # ── Overview table ────────────────────────────────────────────────────
        with st.expander("Overview tutti i frame", expanded=False):
            import pandas as pd
            rows = _build_frame_table(debug)
            df = pd.DataFrame(rows)

            def _hl(row: Any) -> list[str]:
                if row["Frame"] == f:
                    return ["background-color:#fff3cd;font-weight:bold"] * len(row)
                if row["Type"] == "CZ":
                    return ["background-color:#fde8e8"] * len(row)
                return [""] * len(row)

            st.dataframe(df.style.apply(_hl, axis=1), use_container_width=True, height=300)

    _viz()

if __name__ == "__main__" or True:
    main()
