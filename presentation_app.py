"""
Quantum Compilation Presentation App
=====================================
Streamlit interactive explorer for zoned neutral-atom compilation.
"""

from __future__ import annotations

import io
import hashlib
import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

def _persistent_media_cache_dir() -> Path:
    cache_dir = Path(__file__).parent / ".presentation_media_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def _persistent_media_cache_key(kind: str, **parts: Any) -> str:
    payload = {"version": 1, "kind": kind, **parts}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _persistent_media_read(cache_key: str, ext: str) -> bytes | None:
    path = _persistent_media_cache_dir() / f"{cache_key}{ext}"
    try:
        if path.exists(): return path.read_bytes()
    except OSError: return None
    return None

def _persistent_media_write(cache_key: str, ext: str, data: bytes) -> None:
    cache_dir = _persistent_media_cache_dir()
    target = cache_dir / f"{cache_key}{ext}"
    if target.exists(): return
    tmp = cache_dir / f"{cache_key}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_bytes(data)
        tmp.replace(target)
    except OSError:
        try: tmp.unlink(missing_ok=True)
        except OSError: pass

sys.path.insert(0, str(Path(__file__).parent / "python"))
import mqt.qmap  # noqa: F401
import mqt as _mqt_ns
_local_mqt = str(Path(__file__).parent / "python" / "mqt")
if _local_mqt not in list(_mqt_ns.__path__):
    _mqt_ns.__path__.insert(0, _local_mqt)

st.set_page_config(page_title="Quantum Compilation Explorer", layout="wide", initial_sidebar_state="expanded")

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

PRESETS: dict[str, Any] = {
    "CNOT chain (5q)": {"description": "Catena di CNOT", "builder": "cnot_chain", "n_qubits": 5},
    "Teleportation-like (4q)": {"description": "Circuito misto", "builder": "teleport_like", "n_qubits": 4},
    "Linear CZ chain (8q)": {"description": "Catena lineare CZ", "builder": "linear_chain", "n_qubits": 8},
    "GHZ (6q)": {"description": "GHZ state preparation", "builder": "ghz", "n_qubits": 6},
    "QFT (8q)": {"description": "QFT 8 qubit", "builder": "qft", "n_qubits": 8},
    "QFT large (18q)": {"description": "QFT 18 qubit", "builder": "qft", "n_qubits": 18},
    "QPE phase-Z (10q)": {"description": "QPE 9+1 qubit", "builder": "qpe", "n_qubits": 10, "counting_qubits": 9},
    "Dense CZ grid (12q)": {"description": "Stress test router", "builder": "dense_grid", "n_qubits": 12},
    "Brickwork large (24q, 4 layers)": {"description": "Preset corposo", "builder": "brickwork", "n_qubits": 24, "n_layers": 4},
    "Custom QASM": {"description": "Scrivi il tuo OpenQASM 3", "builder": "qasm"},
}

def _build_circuit(preset_name: str, preset: dict[str, Any], qasm_text: str):
    import mqt.core.ir as ir
    import math as _math
    b = preset["builder"]
    n = preset.get("n_qubits", 0)
    if b == "linear_chain":
        qc = ir.QuantumComputation(n)
        for i in range(n - 1): qc.cz(i, i+1)
        return qc
    if b == "ghz":
        qc = ir.QuantumComputation(n)
        qc.h(0)
        for i in range(n - 1): qc.cx(i, i+1)
        return qc
    if b == "qft":
        qc = ir.QuantumComputation(n)
        for i in range(n):
            qc.h(i)
            for j in range(i+1, n): qc.cp(_math.pi / (2**(j-i)), i, j)
        return qc
    if b == "cnot_chain":
        qc = ir.QuantumComputation(n)
        qc.h(0)
        for i in range(n-1): qc.cx(i, i+1)
        return qc
    if b == "qasm":
        return ir.QuantumComputation.from_qasm_str(qasm_text)
    qc = ir.QuantumComputation(n if n > 0 else 4)
    for i in range(qc.num_qubits - 1): qc.cz(i, i+1)
    return qc

@st.cache_data(show_spinner=False)
def compile_circuit(preset_name: str, qasm_text: str, arch_json: str) -> dict[str, Any]:
    import mqt.qmap.na.zoned as zoned
    from mqt.core import load
    from mqt.core.plugins.qiskit import mqt_to_qiskit
    from qiskit import transpile, qasm2
    arch = zoned.ZonedNeutralAtomArchitecture.from_json_string(arch_json)
    qc_logical = _build_circuit(preset_name, PRESETS[preset_name], qasm_text)
    try: original_qasm = qasm2.dumps(mqt_to_qiskit(qc_logical))
    except: original_qasm = ""
    _t1 = transpile(mqt_to_qiskit(qc_logical), basis_gates=["h", "p", "cz"], optimization_level=3)
    qc_native = load(_t1)
    c = zoned.RoutingAwareCompiler(arch)
    c.compile(qc_native)
    di = c.debug_info()
    try: native_qasm = qasm2.dumps(_t1)
    except: native_qasm = ""
    return {"debug": di, "original_qasm": original_qasm, "native_qasm": native_qasm, "n_qubits": di["n_qubits"]}


@st.cache_data(show_spinner=False)
def compute_state_evolution(native_qasm: str, n_qubits: int) -> list:
    """Return statevector per layer as list of {basis, re, im} dicts.
    Only for n_qubits <= 12. Returns [] on error or if too large."""
    if not native_qasm or n_qubits > 12:
        return []
    try:
        from qiskit import qasm2 as _q2, QuantumCircuit
        from qiskit.quantum_info import Statevector
        from qiskit.converters import circuit_to_dag
        from qiskit.dagcircuit import DAGOpNode

        qc = _q2.loads(native_qasm)
        n = qc.num_qubits
        dag = circuit_to_dag(qc)
        sv = Statevector.from_label("0" * n)
        THRESHOLD = 0.005

        def _snap(sv_obj):
            amps = []
            for idx, amp in enumerate(sv_obj.data):
                prob = abs(amp) ** 2
                if prob < THRESHOLD:
                    continue
                amps.append({"basis": format(idx, f"0{n}b"),
                              "re": round(float(amp.real), 5),
                              "im": round(float(amp.imag), 5),
                              "prob": round(float(prob), 5)})
            amps.sort(key=lambda x: -x["prob"])
            return amps[:8]  # keep at most 8 terms

        results = [_snap(sv)]  # initial |0...0⟩ state
        for layer_data in dag.layers():
            layer_qc = QuantumCircuit(n)
            has_ops = False
            for node in layer_data["graph"].nodes():
                if not isinstance(node, DAGOpNode):
                    continue
                if node.op.name in ("barrier", "measure", "delay", "reset"):
                    continue
                qubits = [qc.find_bit(q).index for q in node.qargs]
                try:
                    layer_qc.append(node.op, qubits)
                    has_ops = True
                except Exception:
                    pass
            if not has_ops:
                continue
            sv = sv.evolve(layer_qc)
            results.append(_snap(sv))
        return results
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def render_circuit_image(original_qasm: str) -> bytes | None:
    if not original_qasm:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from qiskit import qasm2 as _q2
        from qiskit.visualization import circuit_drawer
        qc = _q2.loads(original_qasm)
        fig = circuit_drawer(qc, output="mpl", fold=22, initial_state=False)
        fig.patch.set_facecolor("#0d1117")
        for ax in fig.get_axes():
            ax.set_facecolor("#0d1117")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def render_realtime_canvas(debug_json: str, arch_json: str,
                           state_json: str = "[]", height: int = 860) -> None:
    uid = uuid.uuid4().hex[:8]
    debug_meta = json.loads(debug_json)
    n_layers = debug_meta.get("n_layers", 1)
    canvas_h = max(260, height - 288)

    html = f"""
<style>
  #root_{uid}:fullscreen, #root_{uid}:-webkit-full-screen {{
    background: #0a0e1a !important;
    border-radius: 0 !important;
    padding: 16px !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
  }}
</style>
<div id="root_{uid}" style="background:#0a0e1a;border-radius:14px;padding:16px;font-family:'Inter',monospace,system-ui;border:1px solid rgba(255,255,255,0.07);box-shadow:0 8px 32px rgba(0,0,0,0.5);box-sizing:border-box;">

  <!-- Header row -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <span style="color:#e2e8f0;font-weight:700;font-size:0.85rem;letter-spacing:0.08em;text-transform:uppercase;">Atom Array</span>
      <span id="layerctr_{uid}" style="color:#334155;font-size:0.68rem;font-weight:700;letter-spacing:0.06em;">LAYER 0 / {n_layers - 1}</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="display:flex;align-items:center;gap:4px;color:#334155;font-size:0.63rem;"><span style="display:inline-block;width:8px;height:8px;border-radius:1px;background:#2563eb;flex-shrink:0;"></span>1Q</span>
      <span style="display:flex;align-items:center;gap:4px;color:#334155;font-size:0.63rem;"><span style="display:inline-block;width:8px;height:8px;border-radius:1px;background:#9333ea;flex-shrink:0;"></span>2Q</span>
      <span style="display:flex;align-items:center;gap:4px;color:#334155;font-size:0.63rem;"><span style="display:inline-block;width:14px;height:0;border-top:2px dashed rgba(255,255,255,0.2);flex-shrink:0;margin-top:1px;"></span>shuttle</span>
      <span id="phase_{uid}" style="font-size:0.68rem;font-weight:700;padding:3px 12px;border-radius:20px;border:1px solid #3b82f6;background:rgba(59,130,246,0.12);color:#60a5fa;letter-spacing:0.08em;transition:all 0.3s;">READY</span>
      <button id="recbtn_{uid}" title="Record canvas as WebM" style="padding:3px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.05);color:#64748b;cursor:pointer;font-size:0.68rem;font-weight:700;letter-spacing:0.04em;">&#9679; REC</button>
      <button id="fsbtn_{uid}" title="Toggle fullscreen" style="padding:3px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.05);color:#64748b;cursor:pointer;font-size:0.68rem;font-weight:700;letter-spacing:0.04em;">&#x26F6; FULL</button>
    </div>
  </div>

  <!-- Main row: Canvas + Circuit schedule -->
  <div id="mainrow_{uid}" style="display:flex;gap:8px;height:{canvas_h}px;">
    <div style="flex:0 0 58%;position:relative;background:#060b14;border-radius:8px;overflow:hidden;border:1px solid rgba(255,255,255,0.04);">
      <canvas id="canvas_{uid}" style="width:100%;height:100%;display:block;"></canvas>
    </div>
    <!-- NAQC Circuit Schedule panel -->
    <div id="circpanel_{uid}" style="flex:1;background:#060b14;border-radius:8px;overflow:auto;border:1px solid rgba(255,255,255,0.04);position:relative;">
      <div style="position:sticky;top:0;left:0;padding:5px 10px;background:rgba(6,11,20,0.95);z-index:2;border-bottom:1px solid rgba(255,255,255,0.04);">
        <span style="color:#334155;font-size:0.62rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">NAQC Schedule</span>
      </div>
    </div>
  </div>

  <!-- Quantum state evolution panel -->
  <div style="margin-top:8px;padding:8px 14px;background:rgba(255,255,255,0.025);border-radius:6px;min-height:50px;">
    <div style="color:#334155;font-size:0.60rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:3px;">Quantum State</div>
    <div id="statetext_{uid}" style="color:#e2e8f0;font-size:0.78rem;font-family:monospace;line-height:1.6;white-space:pre-wrap;">|0&#x27E9;</div>
  </div>

  <!-- Step description panel -->
  <div style="margin-top:6px;padding:8px 14px;background:rgba(255,255,255,0.025);border-radius:6px;min-height:44px;display:flex;flex-direction:column;justify-content:center;gap:3px;">
    <div id="stepdesc_{uid}" style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">&mdash;</div>
    <div id="stepgates_{uid}" style="color:#475569;font-size:0.67rem;font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">&mdash;</div>
  </div>

  <!-- Segmented progress bar with rugs -->
  <div id="pbwrap_{uid}" style="margin-top:8px;cursor:pointer;border-radius:4px;overflow:hidden;">
    <canvas id="pb_{uid}" style="width:100%;height:34px;display:block;cursor:pointer;"></canvas>
  </div>

  <!-- Controls row -->
  <div style="display:flex;align-items:center;gap:8px;margin-top:8px;background:rgba(255,255,255,0.025);padding:10px 14px;border-radius:8px;">
    <button id="prevbtn_{uid}" title="Previous step" style="padding:6px 14px;border-radius:6px;border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.05);color:#64748b;cursor:pointer;font-weight:700;font-size:0.75rem;letter-spacing:0.04em;transition:background 0.15s;">&#9664; PREV</button>
    <button id="btn_{uid}" style="padding:7px 24px;border-radius:6px;border:none;background:linear-gradient(135deg,#3b82f6,#6366f1);color:white;cursor:pointer;font-weight:700;font-size:0.8rem;min-width:88px;letter-spacing:0.05em;box-shadow:0 2px 8px rgba(99,102,241,0.35);">PLAY</button>
    <button id="nextbtn_{uid}" title="Next step" style="padding:6px 14px;border-radius:6px;border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.05);color:#64748b;cursor:pointer;font-weight:700;font-size:0.75rem;letter-spacing:0.04em;transition:background 0.15s;">NEXT &#9654;</button>
    <div style="flex:1;"></div>
    <span style="color:#334155;font-size:0.68rem;font-weight:700;letter-spacing:0.05em;">SPD</span>
    <select id="speed_{uid}" style="background:#0f1829;color:#94a3b8;border:1px solid rgba(255,255,255,0.10);border-radius:4px;padding:3px 10px;font-size:0.72rem;cursor:pointer;">
      <option value="0.4">0.4×</option>
      <option value="0.7">0.7×</option>
      <option value="1" selected>1×</option>
      <option value="1.8">1.8×</option>
      <option value="3">3×</option>
    </select>
  </div>
</div>

<script>
(() => {{
  const debug     = {debug_json};
  const arch      = {arch_json};
  const stateData = {state_json};

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const canvas      = document.getElementById('canvas_{uid}');
  const ctx         = canvas.getContext('2d');
  const pbCanvas    = document.getElementById('pb_{uid}');
  const pbCtx       = pbCanvas.getContext('2d');
  const pbWrap      = document.getElementById('pbwrap_{uid}');
  const btnEl       = document.getElementById('btn_{uid}');
  const prevEl      = document.getElementById('prevbtn_{uid}');
  const nextEl      = document.getElementById('nextbtn_{uid}');
  const phaseEl     = document.getElementById('phase_{uid}');
  const layerctrEl  = document.getElementById('layerctr_{uid}');
  const stepdescEl  = document.getElementById('stepdesc_{uid}');
  const stepgatesEl = document.getElementById('stepgates_{uid}');
  const speedEl     = document.getElementById('speed_{uid}');
  const circpanel   = document.getElementById('circpanel_{uid}');
  const statetextEl = document.getElementById('statetext_{uid}');
  const recbtnEl    = document.getElementById('recbtn_{uid}');

  // ── Parse architecture ────────────────────────────────────────────────────
  const slmMap = {{}};
  const zones  = [];
  function addZone(z, type) {{
    zones.push({{ type, x:z.offset[0], y:z.offset[1], w:z.dimension[0], h:z.dimension[1] }});
    z.slms.forEach(s => {{
      slmMap[s.id] = {{ x:s.location[0], y:s.location[1], sx:s.site_separation[0], sy:s.site_separation[1], r:s.r, c:s.c }};
    }});
  }}
  (arch.storage_zones      || []).forEach(z => addZone(z, 'storage'));
  (arch.entanglement_zones || []).forEach(z => addZone(z, 'entanglement'));
  const rydRanges = (arch.rydberg_range || []).map(r => ({{ x:r[0][0], y:r[0][1], w:r[1][0]-r[0][0], h:r[1][1]-r[0][1] }}));
  const archRange = arch.arch_range || [[-5,-5],[105,60]];
  const archW = archRange[1][0] - archRange[0][0];
  const archH = archRange[1][1] - archRange[0][1];

  // ── Qubit colour palette ──────────────────────────────────────────────────
  const QC = [
    [96,165,250],[52,211,153],[248,113,113],[251,191,36],[167,139,250],
    [56,189,248],[251,146,60],[163,230,53],[232,121,249],[250,204,21],
    [244,114,182],[45,212,191],[129,140,248],[134,239,172],[252,165,165]
  ];
  function qrgb(i) {{ return QC[i % QC.length]; }}
  function qc(i, a) {{ const [r,g,b] = qrgb(i); return `rgba(${{r}},${{g}},${{b}},${{a}})`; }}

  // ── State ─────────────────────────────────────────────────────────────────
  const totalSteps = 2 * debug.n_layers;
  let progress = 0, playing = false, speed = 1.0, lastTs = null, playUntil = null;

  // ── Layout helpers (CSS-px space; DPR via ctx.scale) ─────────────────────
  let dpr=1, cssW=0, cssH=0, scl=1, tx=0, ty=0;
  function resize() {{
    dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    cssW = rect.width; cssH = rect.height;
    canvas.width  = cssW * dpr; canvas.height = cssH * dpr;
    canvas.style.width  = cssW + 'px'; canvas.style.height = cssH + 'px';
    const pad = 18;
    scl = Math.min((cssW - 2*pad) / archW, (cssH - 2*pad) / archH);
    tx  = (cssW - archW*scl)/2 - archRange[0][0]*scl;
    ty  = (cssH - archH*scl)/2 - archRange[0][1]*scl;
  }}
  const X  = wx => wx*scl + tx;
  const Y  = wy => wy*scl + ty;
  const Sp = ws => ws*scl;

  // ── Phase timing ──────────────────────────────────────────────────────────
  // 0.00 – 0.32 : gate fully visible, qubits stationary
  // 0.32 – 0.50 : gate fades out
  // 0.50 – 1.00 : qubits move (ease-in-out), gate gone
  function gateAlpha(t) {{
    if (t <= 0.32) return 1.0;
    if (t <  0.50) return (0.50 - t) / 0.18;
    return 0;
  }}
  function moveT(t) {{
    if (t <= 0.50) return 0;
    const u = (t - 0.50) / 0.50;
    return u < 0.5 ? 2*u*u : 1 - Math.pow(-2*u+2, 2)/2;
  }}

  // ── Qubit world-position lookup ───────────────────────────────────────────
  function qpos(qi, frame) {{
    const p = debug.placement[Math.min(frame, debug.placement.length - 1)];
    if (!p || p[qi] == null) return null;
    const [sid, row, col] = p[qi];
    const s = slmMap[sid]; if (!s) return null;
    return {{ wx: s.x + col*s.sx, wy: s.y + row*s.sy }};
  }}

  // ── Rounded rect helper ───────────────────────────────────────────────────
  function rrect(c, x, y, w, h, r) {{
    if (c.roundRect) {{ c.beginPath(); c.roundRect(x,y,w,h,r); return; }}
    c.beginPath();
    c.moveTo(x+r,y); c.lineTo(x+w-r,y); c.quadraticCurveTo(x+w,y,x+w,y+r);
    c.lineTo(x+w,y+h-r); c.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
    c.lineTo(x+r,y+h); c.quadraticCurveTo(x,y+h,x,y+h-r);
    c.lineTo(x,y+r); c.quadraticCurveTo(x,y,x+r,y);
    c.closePath();
  }}

  // ── Progress bar with rugs ────────────────────────────────────────────────
  function drawPB() {{
    const dpr2   = window.devicePixelRatio || 1;
    const cssWPB = pbWrap.getBoundingClientRect().width || 100;
    const cssHPB = 34;
    pbCanvas.width  = cssWPB * dpr2;
    pbCanvas.height = cssHPB * dpr2;
    pbCanvas.style.width  = cssWPB + 'px';
    pbCanvas.style.height = cssHPB + 'px';
    pbCtx.clearRect(0, 0, pbCanvas.width, pbCanvas.height);
    pbCtx.save();
    pbCtx.scale(dpr2, dpr2);

    const W = cssWPB, H = cssHPB;
    // Background
    pbCtx.fillStyle = '#0c1220';
    rrect(pbCtx, 0, 0, W, H, 4); pbCtx.fill();

    const segW = W / totalSteps;
    const barY = 10, barH = 14;

    // Colour segments: blue=1Q, purple=2Q; brighter=current/past
    for (let i = 0; i < totalSteps; i++) {{
      const isE  = (i % 2) === 1;
      const cur  = Math.floor(progress) === i;
      const past = i < progress;
      if (isE) {{
        pbCtx.fillStyle = cur ? '#9333ea' : (past ? 'rgba(147,51,234,0.55)' : 'rgba(147,51,234,0.18)');
      }} else {{
        pbCtx.fillStyle = cur ? '#2563eb' : (past ? 'rgba(37,99,235,0.55)' : 'rgba(37,99,235,0.18)');
      }}
      pbCtx.fillRect(i*segW + 0.5, barY, segW - 1, barH);
    }}

    // Rugs: white ticks at layer boundaries (every 2 steps = 1 layer)
    for (let i = 0; i <= totalSteps; i += 2) {{
      const x = i * segW;
      pbCtx.strokeStyle = 'rgba(255,255,255,0.30)';
      pbCtx.lineWidth = 1;
      pbCtx.beginPath(); pbCtx.moveTo(x, barY - 4); pbCtx.lineTo(x, barY + barH + 4); pbCtx.stroke();
      if (i < totalSteps) {{
        pbCtx.fillStyle = 'rgba(255,255,255,0.35)';
        pbCtx.font = '7px monospace';
        pbCtx.textAlign = 'left';
        pbCtx.fillText('L' + (i/2), x + 1.5, barY - 5);
      }}
    }}

    // Needle (current position)
    const nx = (progress / totalSteps) * W;
    pbCtx.fillStyle = 'rgba(255,255,255,0.92)';
    pbCtx.fillRect(nx - 1, barY - 5, 2, barH + 10);

    // Step counter near needle
    const stepLabel = (Math.floor(progress) + 1) + '/' + totalSteps;
    pbCtx.font = 'bold 8px monospace';
    pbCtx.fillStyle = 'rgba(255,255,255,0.65)';
    const labRight = nx > W * 0.80;
    pbCtx.textAlign = labRight ? 'right' : 'left';
    pbCtx.fillText(stepLabel, nx + (labRight ? -4 : 4), barY + barH + 11);

    pbCtx.restore();
  }}

  // ── Step description panel ────────────────────────────────────────────────
  function updateStepInfo(f0) {{
    const isEnt = (f0 % 2) === 1;
    const layer = Math.floor(f0 / 2);
    const col   = isEnt ? '#c084fc' : '#60a5fa';

    stepdescEl.innerHTML =
      `<span style="color:${{col}};font-weight:700;">Layer ${{layer}}</span>` +
      `<span style="color:#1e293b;"> &mdash; </span>` +
      `<span style="color:#94a3b8;">${{isEnt ? 'Rydberg Entanglement' : 'Single-Qubit Gates'}}</span>`;

    let gatesHtml = '';
    if (isEnt) {{
      const pairs = debug.two_qubit_layers[layer] || [];
      gatesHtml = pairs.map(p =>
        `<span style="color:#a78bfa;margin-right:12px;">CZ(q${{p[0]}},&#8202;q${{p[1]}})</span>`
      ).join('');
    }} else {{
      const ops = debug.single_qubit_layers[layer] || [];
      const opByQ = {{}};
      ops.forEach(op => (op.qubits || []).forEach(q => {{ opByQ[q] = (op.name || '1Q').toUpperCase(); }}));
      const qubits = Object.keys(opByQ).map(Number).sort((a,b) => a-b);
      gatesHtml = qubits.map(q =>
        `<span style="color:#67e8f9;margin-right:12px;">${{opByQ[q]}}(q${{q}})</span>`
      ).join('');
    }}
    stepgatesEl.innerHTML = gatesHtml || '<span style="color:#1e293b;">—</span>';

    // Phase badge
    phaseEl.textContent = isEnt ? 'RYDBERG CZ' : '1Q GATES';
    Object.assign(phaseEl.style, isEnt
      ? {{ background:'rgba(168,85,247,0.18)', color:'#c084fc', borderColor:'#a855f7' }}
      : {{ background:'rgba(59,130,246,0.14)', color:'#60a5fa', borderColor:'#3b82f6' }}
    );
    layerctrEl.textContent = `LAYER ${{layer}} / ${{debug.n_layers - 1}}`;
  }}

  // ── Pre-process routing waypoints from operations ─────────────────────────
  // Between each pair of CZ operations there are TWO routing rounds:
  //   1) "going-back"  – atoms returning from EZ to storage
  //   2) "going-forward" – atoms leaving storage toward the next EZ
  // transIdx must increment on EACH completed round, not only on CZ.
  const routingPaths = (() => {{
    const ops = debug.operations || [];
    const nTrans = (debug.placement || []).length - 1;
    const paths = {{}};
    let transIdx = 0;
    const qubitPath = {{}};
    let roundComplete = false; // true after every store empties qubitPath
    for (const op of ops) {{
      if (op.type === 'load') {{
        // A new round starts after a completed round (going-back → going-forward boundary)
        if (roundComplete && Object.keys(qubitPath).length === 0) {{
          transIdx++;
          roundComplete = false;
        }}
        for (const qi of (op.qubits || [])) qubitPath[qi] = {{ transIdx, waypoints: [] }};
      }} else if (op.type === 'move') {{
        const qubits = op.qubits || [], targets = op.targets || [];
        for (let k = 0; k < qubits.length; k++) {{
          const qi = qubits[k];
          if (k < targets.length && qubitPath[qi]) qubitPath[qi].waypoints.push(targets[k]);
        }}
      }} else if (op.type === 'store') {{
        for (const qi of (op.qubits || [])) {{
          if (qubitPath[qi]) {{
            const ti = qubitPath[qi].transIdx;
            if (!paths[ti]) paths[ti] = {{}};
            paths[ti][qi] = qubitPath[qi].waypoints;
            delete qubitPath[qi];
          }}
        }}
        // Mark round complete when all active paths have been stored
        if (Object.keys(qubitPath).length === 0) roundComplete = true;
      }} else if (op.type === 'cz') {{
        // CZ always advances to the next transition; reset round tracking
        transIdx++;
        roundComplete = false;
      }}
    }}
    const result = [];
    for (let i = 0; i < nTrans; i++) result.push(paths[i] || {{}});
    return result;
  }})();

  // ── Pre-build reuse annotations: reuseAnnotations[f0] = Set of qubit indices ──
  const reuseAnnotations = {{}};
  (debug.reuse_qubits || []).forEach((set, i) => {{
    const f0 = 2 * (i + 1); // even step (1Q phase) after CZ layer i
    if (!reuseAnnotations[f0]) reuseAnnotations[f0] = new Set();
    set.forEach(q => reuseAnnotations[f0].add(q));
  }});

  // ── Main draw ─────────────────────────────────────────────────────────────
  function draw() {{
    const f0    = Math.min(Math.floor(progress), totalSteps - 1);
    const t     = progress - f0;
    const isEnt = (f0 % 2) === 1;
    const layer = Math.floor(f0 / 2);
    const ga    = gateAlpha(t);
    const mt    = moveT(t);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#060b14';
    ctx.fillRect(0, 0, cssW, cssH);

    // Rydberg range dashed outline
    if (isEnt && ga > 0.01) {{
      rydRanges.forEach(r => {{
        ctx.save();
        ctx.globalAlpha = ga * 0.06;
        ctx.fillStyle = '#a855f7';
        rrect(ctx, X(r.x), Y(r.y), Sp(r.w), Sp(r.h), 5); ctx.fill();
        ctx.globalAlpha = ga * 0.28;
        ctx.strokeStyle = '#a855f7'; ctx.lineWidth = 1; ctx.setLineDash([5,5]);
        rrect(ctx, X(r.x), Y(r.y), Sp(r.w), Sp(r.h), 5); ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }});
    }}

    // Zones
    zones.forEach(z => {{
      const zx=X(z.x-2), zy=Y(z.y-2), zw=Sp(z.w+4), zh=Sp(z.h+4);
      const isE = z.type === 'entanglement', active = isEnt && isE;
      ctx.save();
      ctx.globalAlpha = active ? 0.16 : 0.07;
      ctx.fillStyle = isE ? '#a855f7' : '#3b82f6';
      rrect(ctx, zx, zy, zw, zh, 5); ctx.fill();
      ctx.globalAlpha = active ? 0.55 : 0.22;
      ctx.strokeStyle = isE ? '#a855f7' : '#3b82f6'; ctx.lineWidth = 1;
      rrect(ctx, zx, zy, zw, zh, 5); ctx.stroke();
      ctx.globalAlpha = 0.38;
      ctx.fillStyle = isE ? '#c084fc' : '#60a5fa';
      ctx.font = `bold ${{Math.round(Sp(2.0))}}px monospace`;
      ctx.fillText(isE ? 'ENT' : 'STO', X(z.x), Y(z.y + z.h) + Sp(2.8));
      ctx.restore();
    }});

    // SLM trap sites
    ctx.fillStyle = 'rgba(255,255,255,0.09)';
    Object.values(slmMap).forEach(s => {{
      for (let r=0; r<s.r; r++) for (let c=0; c<s.c; c++) {{
        ctx.beginPath();
        ctx.arc(X(s.x+c*s.sx), Y(s.y+r*s.sy), Math.max(1.0, Sp(0.28)), 0, Math.PI*2);
        ctx.fill();
      }}
    }});

    // Interpolated qubit positions (with routing waypoints)
    const coords = [];
    const f1 = Math.min(f0 + 1, totalSteps);
    for (let i = 0; i < debug.n_qubits; i++) {{
      const p0 = qpos(i, f0);
      if (!p0) {{ coords.push(null); continue; }}
      const p1 = qpos(i, f1) || p0;
      const waypts = (routingPaths[f0] && routingPaths[f0][i]) || null;
      let wx, wy;
      if (mt <= 0 || !waypts || waypts.length === 0) {{
        wx = p0.wx + (p1.wx - p0.wx) * mt;
        wy = p0.wy + (p1.wy - p0.wy) * mt;
      }} else {{
        // Animate along the full waypoint path: start → wp[0] → … → wp[n-1]
        const path = [[p0.wx, p0.wy], ...waypts];
        const nSeg = path.length - 1;
        const segProg = mt * nSeg;
        const si = Math.min(Math.floor(segProg), nSeg - 1);
        const st = segProg - si;
        wx = path[si][0] + (path[si+1][0] - path[si][0]) * st;
        wy = path[si][1] + (path[si+1][1] - path[si][1]) * st;
      }}
      coords.push({{ wx, wy, wx0:p0.wx, wy0:p0.wy, wx1:p1.wx, wy1:p1.wy, waypts }});
    }}

    // Movement trails (with waypoint paths)
    if (mt > 0.02 && mt < 0.98) {{
      coords.forEach((c, i) => {{
        if (!c) return;
        const moved = Math.abs(c.wx1-c.wx0) > 0.05 || Math.abs(c.wy1-c.wy0) > 0.05;
        if (!moved) return;
        ctx.save();
        if (c.waypts && c.waypts.length > 0) {{
          const path = [[c.wx0, c.wy0], ...c.waypts];
          const nSeg = path.length - 1;
          // Full ghost path (dim dashed)
          ctx.globalAlpha = 0.16; ctx.strokeStyle = qc(i,1); ctx.lineWidth = 1;
          ctx.setLineDash([3,4]);
          ctx.beginPath(); ctx.moveTo(X(path[0][0]), Y(path[0][1]));
          for (let k = 1; k < path.length; k++) ctx.lineTo(X(path[k][0]), Y(path[k][1]));
          ctx.stroke(); ctx.setLineDash([]);
          // Completed portion (brighter solid)
          const segProg = mt * nSeg;
          const si = Math.min(Math.floor(segProg), nSeg - 1);
          ctx.globalAlpha = 0.50; ctx.strokeStyle = qc(i,1); ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(X(path[0][0]), Y(path[0][1]));
          for (let k = 1; k <= si; k++) ctx.lineTo(X(path[k][0]), Y(path[k][1]));
          ctx.lineTo(X(c.wx), Y(c.wy)); ctx.stroke();
          // Waypoint dots
          ctx.globalAlpha = 0.35; ctx.fillStyle = qc(i, 1);
          for (let k = 1; k < path.length - 1; k++) {{
            ctx.beginPath(); ctx.arc(X(path[k][0]), Y(path[k][1]), 2.5, 0, Math.PI*2); ctx.fill();
          }}
        }} else {{
          ctx.globalAlpha = 0.28; ctx.strokeStyle = qc(i,1); ctx.lineWidth = 1;
          ctx.setLineDash([3,4]);
          ctx.beginPath(); ctx.moveTo(X(c.wx0),Y(c.wy0)); ctx.lineTo(X(c.wx),Y(c.wy)); ctx.stroke();
          ctx.setLineDash([]);
        }}
        ctx.globalAlpha = 0.11; ctx.strokeStyle = qc(i,1); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(X(c.wx1),Y(c.wy1), Math.max(3.5, Sp(1.5)), 0, Math.PI*2); ctx.stroke();
        ctx.restore();
      }});
    }}

    // 1Q laser gates
    if (!isEnt && ga > 0.01) {{
      (debug.single_qubit_layers[layer] || []).forEach(op => {{
        (op.qubits || []).forEach(qi => {{
          const c = coords[qi]; if (!c) return;
          const cx=X(c.wx), cy=Y(c.wy);
          ctx.save();
          ctx.globalAlpha = ga * 0.28; ctx.strokeStyle = '#67e8f9'; ctx.lineWidth = 0.5; ctx.setLineDash([5,7]);
          ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,cssH); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(0,cy); ctx.lineTo(cssW,cy); ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = ga;
          const gr = Sp(5.5);
          const g = ctx.createRadialGradient(cx,cy,0,cx,cy,gr);
          g.addColorStop(0,    `rgba(103,232,249,${{0.80*ga}})`);
          g.addColorStop(0.45, `rgba(103,232,249,${{0.20*ga}})`);
          g.addColorStop(1,    'rgba(103,232,249,0)');
          ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx,cy,gr,0,Math.PI*2); ctx.fill();
          ctx.restore();
        }});
      }});
    }}

    // 2Q Rydberg gates
    if (isEnt && ga > 0.01) {{
      (debug.two_qubit_layers[layer] || []).forEach(pair => {{
        const c1=coords[pair[0]], c2=coords[pair[1]]; if (!c1||!c2) return;
        const x1=X(c1.wx), y1=Y(c1.wy), x2=X(c2.wx), y2=Y(c2.wy);
        const mx=(x1+x2)/2, my=(y1+y2)/2, dist=Math.hypot(x2-x1,y2-y1);
        ctx.save();
        ctx.globalAlpha = ga*0.18;
        const hr = Math.max(dist*0.55, Sp(7));
        const hg = ctx.createRadialGradient(mx,my,0,mx,my,hr);
        hg.addColorStop(0,'#c084fc'); hg.addColorStop(1,'rgba(192,132,252,0)');
        ctx.fillStyle=hg; ctx.beginPath(); ctx.arc(mx,my,hr,0,Math.PI*2); ctx.fill();
        ctx.globalAlpha=ga;
        const lg=ctx.createLinearGradient(x1,y1,x2,y2);
        lg.addColorStop(0, `rgba(192,132,252,${{ga}})`);
        lg.addColorStop(0.5,`rgba(240,171,252,${{ga}})`);
        lg.addColorStop(1, `rgba(192,132,252,${{ga}})`);
        ctx.strokeStyle=lg; ctx.lineWidth=2;
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
        ctx.globalAlpha=ga*0.65; ctx.strokeStyle='#e879f9'; ctx.lineWidth=1.5;
        [[x1,y1],[x2,y2]].forEach(([px,py]) => {{ ctx.beginPath(); ctx.arc(px,py,Sp(2.6),0,Math.PI*2); ctx.stroke(); }});
        ctx.restore();
      }});
    }}

    // Qubits
    coords.forEach((c, i) => {{
      if (!c) return;
      const cx=X(c.wx), cy=Y(c.wy);
      const r=Math.max(4, Sp(1.5));
      const [cr,cg,cb]=qrgb(i);
      // Outer glow
      ctx.save();
      const gl=ctx.createRadialGradient(cx,cy,0,cx,cy,r*3.5);
      gl.addColorStop(0,`rgba(${{cr}},${{cg}},${{cb}},0.30)`);
      gl.addColorStop(1,`rgba(${{cr}},${{cg}},${{cb}},0)`);
      ctx.fillStyle=gl; ctx.beginPath(); ctx.arc(cx,cy,r*3.5,0,Math.PI*2); ctx.fill();
      ctx.restore();
      // Core
      ctx.save();
      ctx.fillStyle=`rgba(${{cr}},${{cg}},${{cb}},0.92)`;
      ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.fill();
      ctx.strokeStyle='rgba(255,255,255,0.50)'; ctx.lineWidth=0.8; ctx.stroke();
      ctx.restore();
      // Label
      ctx.save();
      ctx.fillStyle=`rgba(${{cr}},${{cg}},${{cb}},0.88)`;
      ctx.font=`bold ${{Math.max(9,Math.round(Sp(1.7)))}}px monospace`;
      ctx.fillText(`q${{i}}`, cx+r+2, cy+3.5);
      ctx.restore();
    }});

    // Reuse annotations: small "♻ reuse" badge above qubits staying in EZ
    const reuseSet = reuseAnnotations[f0];
    if (reuseSet && reuseSet.size > 0) {{
      reuseSet.forEach(qi => {{
        const c = coords[qi]; if (!c) return;
        const cx = X(c.wx), cy = Y(c.wy);
        const r = Math.max(4, Sp(1.5));
        const bw = 54, bh = 15, bx = cx - bw/2, by = cy - r - bh - 5;
        ctx.save();
        ctx.fillStyle = 'rgba(20,83,45,0.88)';
        rrect(ctx, bx, by, bw, bh, 3); ctx.fill();
        ctx.strokeStyle = '#4ade80'; ctx.lineWidth = 0.8;
        rrect(ctx, bx, by, bw, bh, 3); ctx.stroke();
        ctx.fillStyle = '#4ade80';
        ctx.font = `bold ${{Math.max(8, Math.round(Sp(1.6)))}}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(`\u267b q${{qi}} reuse`, cx, by + bh - 3.5);
        ctx.restore();
      }});
    }}

    ctx.restore(); // undo DPR scale

    updateStepInfo(f0);
    drawPB();
    updateCircuitSVG(f0);
    updateStatePanel(f0);
  }}

  // ── Animation loop ────────────────────────────────────────────────────────
  function animate(ts) {{
    if (!playing) return;
    if (lastTs != null) {{
      const dt = Math.min((ts - lastTs) / 1000, 0.05);
      progress += dt * speed * 0.75;
      if (playUntil !== null && progress >= playUntil) {{
        progress = playUntil;
        playUntil = null;
        playing = false;
        btnEl.textContent = 'PLAY';
        draw();
        return;
      }}
      if (progress >= totalSteps) progress = 0; // loop
    }}
    lastTs = ts;
    draw();
    if (playing) requestAnimationFrame(animate);
  }}

  // ── Controls ──────────────────────────────────────────────────────────────
  btnEl.onclick = () => {{
    playing = !playing;
    btnEl.textContent = playing ? 'PAUSE' : 'PLAY';
    if (playing) {{ lastTs = null; requestAnimationFrame(animate); }}
    else draw();
  }};

  prevEl.onclick = () => {{
    playing = false; btnEl.textContent = 'PLAY';
    const cur = Math.floor(progress);
    progress = (progress - cur < 0.05) ? Math.max(0, cur - 1) : cur;
    draw();
  }};

  nextEl.onclick = () => {{
    if (progress >= totalSteps - 0.001) return;
    playUntil = Math.min(totalSteps - 0.001, Math.floor(progress) + 1);
    if (!playing) {{
      playing = true; btnEl.textContent = 'PAUSE'; lastTs = null;
      requestAnimationFrame(animate);
    }}
  }};

  speedEl.onchange = () => {{ speed = parseFloat(speedEl.value); }};

  // Click on progress bar → jump to step
  pbCanvas.onclick = e => {{
    const rect = pbCanvas.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    progress = Math.max(0, Math.min(totalSteps - 0.001, frac * totalSteps));
    if (!playing) draw(); else drawPB();
  }};

  // ── NAQC Circuit Schedule SVG ─────────────────────────────────────────────
  let lastCircStep = -1;
  function updateCircuitSVG(f0) {{
    if (f0 === lastCircStep) return;
    lastCircStep = f0;
    const panelW = circpanel.clientWidth  || 300;
    const panelH = circpanel.clientHeight || 400;
    const nQ     = debug.n_qubits;
    const nSteps = totalSteps;
    const LPAD = 32, TPAD = 32; // TPAD leaves room for sticky header
    const WIRE_H = Math.max(14, Math.min(34, (panelH - TPAD - 12) / nQ));
    const COL_W  = Math.max(26, Math.min(46, (panelW - LPAD - 8) / nSteps));
    const svgW   = Math.max(panelW, LPAD + nSteps * COL_W + 10);
    const svgH   = TPAD + nQ * WIRE_H + 12;
    const GS     = Math.min(WIRE_H - 4, 18); // gate box size

    let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${{svgW}}" height="${{svgH}}" style="display:block;">`;
    // qubit labels + wires
    for (let q = 0; q < nQ; q++) {{
      const y = TPAD + q * WIRE_H + WIRE_H / 2;
      s += `<text x="${{LPAD-4}}" y="${{y+3.5}}" fill="#334155" font-size="9" text-anchor="end" font-family="monospace">q${{q}}</text>`;
      s += `<line x1="${{LPAD}}" y1="${{y}}" x2="${{svgW-4}}" y2="${{y}}" stroke="#1e293b" stroke-width="0.8"/>`;
    }}
    // column backgrounds + gates
    for (let step = 0; step < nSteps; step++) {{
      const isE   = (step % 2) === 1;
      const layer = Math.floor(step / 2);
      const cx    = LPAD + step * COL_W + COL_W / 2;
      const x0    = LPAD + step * COL_W;
      const isCur = step === f0;
      const isPast= step < f0;
      // background
      if (isCur) {{
        const bg = isE ? 'rgba(147,51,234,0.22)' : 'rgba(37,99,235,0.20)';
        const bc = isE ? '#9333ea' : '#2563eb';
        s += `<rect x="${{x0+1}}" y="${{TPAD-6}}" width="${{COL_W-2}}" height="${{svgH-TPAD+6}}" fill="${{bg}}" stroke="${{bc}}" stroke-width="0.7" rx="2"/>`;
      }} else if (isPast) {{
        s += `<rect x="${{x0+1}}" y="${{TPAD-6}}" width="${{COL_W-2}}" height="${{svgH-TPAD+6}}" fill="rgba(255,255,255,0.02)" rx="2"/>`;
      }}
      // layer boundary tick
      if (step % 2 === 0) {{
        s += `<line x1="${{x0}}" y1="${{TPAD-8}}" x2="${{x0}}" y2="${{svgH-2}}" stroke="rgba(255,255,255,0.06)" stroke-width="0.5"/>`;
        if (COL_W * 2 > 22) s += `<text x="${{x0+1}}" y="${{TPAD-10}}" fill="#1e3050" font-size="7" font-family="monospace">L${{layer}}</text>`;
      }}
      if (isE) {{
        // 2Q Rydberg gates
        (debug.two_qubit_layers[layer] || []).forEach(pair => {{
          const y0 = TPAD + pair[0] * WIRE_H + WIRE_H/2;
          const y1 = TPAD + pair[1] * WIRE_H + WIRE_H/2;
          const fill  = isCur ? '#c084fc' : (isPast ? '#6b21a8' : '#2e1065');
          const lc    = isCur ? '#e879f9' : (isPast ? '#7c3aed' : '#3b0764');
          const sw    = isCur ? 1.8 : 1;
          s += `<line x1="${{cx}}" y1="${{y0}}" x2="${{cx}}" y2="${{y1}}" stroke="${{lc}}" stroke-width="${{sw}}"/>`;
          s += `<circle cx="${{cx}}" cy="${{y0}}" r="${{isCur?5:4}}" fill="${{fill}}" stroke="${{isCur?'#f0abfc':'#581c87'}}" stroke-width="0.8"/>`;
          s += `<circle cx="${{cx}}" cy="${{y1}}" r="${{isCur?5:4}}" fill="${{fill}}" stroke="${{isCur?'#f0abfc':'#581c87'}}" stroke-width="0.8"/>`;
          if (isCur) s += `<text x="${{cx+6}}" y="${{(y0+y1)/2+3}}" fill="#c084fc" font-size="7" font-family="monospace">CZ</text>`;
        }});
      }} else {{
        // 1Q gates
        (debug.single_qubit_layers[layer] || []).forEach(op => {{
          (op.qubits || []).forEach(q => {{
            const y   = TPAD + q * WIRE_H + WIRE_H/2;
            const bf  = isCur ? '#1d4ed8' : (isPast ? '#1e3a5f' : '#0f172a');
            const bst = isCur ? '#3b82f6' : (isPast ? '#1e40af' : '#1e293b');
            const tc  = isCur ? '#bfdbfe' : (isPast ? '#3b82f6' : '#1e3060');
            const nm  = ((op.name || '?').toUpperCase()).substring(0,2);
            s += `<rect x="${{cx-GS/2}}" y="${{y-GS/2}}" width="${{GS}}" height="${{GS}}" fill="${{bf}}" stroke="${{bst}}" stroke-width="0.8" rx="2"/>`;
            s += `<text x="${{cx}}" y="${{y+3.5}}" fill="${{tc}}" font-size="${{Math.max(7,GS*0.45)}}" text-anchor="middle" font-family="monospace" font-weight="bold">${{nm}}</text>`;
          }});
        }});
      }}
    }}
    s += '</svg>';
    // preserve sticky header, replace rest
    const header = circpanel.querySelector('div');
    circpanel.innerHTML = '';
    if (header) circpanel.appendChild(header);
    circpanel.insertAdjacentHTML('beforeend', s);
    // auto-scroll active column into view
    circpanel.scrollLeft = Math.max(0, LPAD + f0 * COL_W - circpanel.clientWidth / 2 + COL_W / 2);
  }}

  // ── Quantum state evolution panel ─────────────────────────────────────────
  let lastStateStep = -1;
  function fmtTerm(t) {{
    const mag = Math.sqrt(t.re * t.re + t.im * t.im);
    if (mag < 0.005) return null;
    const fracs = [[1.0,''], [0.7071,'1/\u221a2\u00b7'], [0.5,'\u00bd\u00b7'],
                   [0.3536,'1/(2\u221a2)\u00b7'], [0.25,'\u00bc\u00b7']];
    let coeff = mag.toFixed(3) + '\u00b7';
    for (const [v, s] of fracs) {{ if (Math.abs(mag - v) < 0.004) {{ coeff = s; break; }} }}
    // phase
    let sign = '+';
    if (Math.abs(t.im) < 0.002) {{
      if (t.re < -0.001) sign = '-';
    }} else {{
      const ang = Math.atan2(t.im, t.re) * 180 / Math.PI;
      if (Math.abs(ang - 180) < 3 || ang < -177) sign = '-';
    }}
    return {{ sign, str: `${{coeff}}|${{t.basis}}\u27e9` }};
  }}
  function updateStatePanel(f0) {{
    if (!stateData || stateData.length === 0) return;
    if (f0 === lastStateStep) return;
    lastStateStep = f0;
    const si = Math.min(stateData.length - 1,
      Math.round(f0 * (stateData.length - 1) / Math.max(totalSteps - 1, 1)));
    const terms = stateData[si] || [];
    if (terms.length === 0) {{ statetextEl.textContent = '|\u2205\u27e9'; return; }}
    const fmt = terms.map(fmtTerm).filter(Boolean);
    if (fmt.length === 0) {{ statetextEl.textContent = '|0\u27e9'; return; }}
    let out = fmt[0].str;
    for (let i = 1; i < fmt.length; i++) out += `  ${{fmt[i].sign}}  ${{fmt[i].str}}`;
    statetextEl.textContent = out;
  }}

  // ── Canvas recorder (WebM download) ──────────────────────────────────────
  let recorder = null, recChunks = [];
  recbtnEl.onclick = () => {{
    if (recorder && recorder.state === 'recording') {{
      recorder.stop();
      recbtnEl.textContent = '\u25cf REC';
      recbtnEl.style.color = '#64748b';
      return;
    }}
    try {{
      const stream = canvas.captureStream(30);
      const mime   = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm']
                       .find(m => MediaRecorder.isTypeSupported(m)) || 'video/webm';
      recorder = new MediaRecorder(stream, {{ mimeType: mime }});
      recChunks = [];
      recorder.ondataavailable = e => {{ if (e.data.size > 0) recChunks.push(e.data); }};
      recorder.onstop = () => {{
        const blob = new Blob(recChunks, {{ type: 'video/webm' }});
        const url  = URL.createObjectURL(blob);
        const a    = Object.assign(document.createElement('a'), {{ href:url, download:'quantum_animation.webm' }});
        a.click();
        URL.revokeObjectURL(url);
        recbtnEl.textContent = '\u25cf REC';
        recbtnEl.style.color = '#64748b';
      }};
      recorder.start(100); // collect chunks every 100ms
      recbtnEl.textContent = '\u23f9 STOP';
      recbtnEl.style.color = '#ef4444';
      // auto-start playback from beginning
      progress = 0;
      if (!playing) {{ playing = true; btnEl.textContent = 'PAUSE'; lastTs = null; requestAnimationFrame(animate); }}
    }} catch(e) {{ recbtnEl.title = 'MediaRecorder not supported: ' + e.message; }}
  }};

  // ── Fullscreen ────────────────────────────────────────────────────────────
  const rootEl   = document.getElementById('root_{uid}');
  const mainRow  = document.getElementById('mainrow_{uid}');
  const fsbtnEl  = document.getElementById('fsbtn_{uid}');
  const NORMAL_H = {canvas_h}; // baseline canvas height in normal mode

  function applyLayout() {{
    const isFull = !!document.fullscreenElement;
    if (isFull) {{
      // Fill the entire screen; subtract fixed chrome (header ~38px, state+desc+pb+controls ~180px, padding 32px)
      const avail = window.innerHeight - 38 - 180 - 32;
      mainRow.style.height = Math.max(200, avail) + 'px';
      fsbtnEl.textContent = '\u2715 EXIT';
    }} else {{
      mainRow.style.height = NORMAL_H + 'px';
      fsbtnEl.textContent = '\u26f6 FULL';
    }}
    resize();
    draw();
  }}

  fsbtnEl.onclick = () => {{
    if (!document.fullscreenElement) {{
      rootEl.requestFullscreen().catch(() => {{}});
    }} else {{
      document.exitFullscreen();
    }}
  }};

  document.addEventListener('fullscreenchange', applyLayout);

  window.addEventListener('resize', () => {{ resize(); draw(); }});
  setTimeout(() => {{ resize(); draw(); }}, 30);
}})();
</script>
"""
    components.html(html, height=height)


def _debug_asm(di: dict) -> str:
    """Render debug_info as compact assembly-like text."""
    lines: list[str] = []
    nq, nl = di["n_qubits"], di["n_layers"]
    lines.append(f"; {nq}q  {nl} layers")
    lines.append("")

    # Gate schedule
    for layer in range(nl):
        sq = di["single_qubit_layers"][layer] if layer < len(di["single_qubit_layers"]) else []
        tq = di["two_qubit_layers"][layer]    if layer < len(di["two_qubit_layers"])    else []
        sq_str = "  ".join(
            f"{op['name'].upper()} q{q}"
            for op in sq for q in op.get("qubits", [])
        ) or "—"
        tq_str = "  ".join(f"CZ q{p[0]},q{p[1]}" for p in tq) or "—"
        reuse = di["reuse_qubits"][layer] if layer < len(di.get("reuse_qubits", [])) else []
        r_str = f"  ; reuse {{{','.join(f'q{q}' for q in reuse)}}}" if reuse else ""
        lines.append(f"L{layer:02d}  1Q: {sq_str}")
        lines.append(f"     2Q: {tq_str}{r_str}")

    # Placement
    lines.append("")
    lines.append("; placement[frame][qubit] = slm:row,col")
    slm_label = {0: "STO", 1: "EZ1", 2: "EZ2"}
    for fi, frame in enumerate(di["placement"]):
        parts = [f"q{qi}→{slm_label.get(sid, f's{sid}')}:{r},{c}"
                 for qi, (sid, r, c) in enumerate(frame)]
        lines.append(f"P{fi:02d}  " + "  ".join(parts))

    # Operations
    lines.append("")
    lines.append("; operations")
    for op in di["operations"]:
        t = op["type"]
        qs = ",".join(f"q{q}" for q in op.get("qubits", []))
        if t == "move":
            tgts = "  ".join(f"({x:.0f},{y:.0f})" for x, y in op.get("targets", []))
            lines.append(f"  {t:<8} {qs}  →  {tgts}")
        elif qs:
            lines.append(f"  {t:<8} {qs}")
        else:
            lines.append(f"  {t}")

    return "\n".join(lines)


def main() -> None:
    st.session_state.setdefault("compiled", False)

    with st.sidebar:
        st.title("Quantum Compiler")
        preset_name = st.selectbox("Circuito", list(PRESETS.keys()))
        preset = PRESETS[preset_name]
        qasm_text = ""
        if preset["builder"] == "qasm":
            qasm_text = st.text_area("OpenQASM 3", height=180)

        if st.button("Compila", type="primary", use_container_width=True):
            with st.spinner("Compilazione..."):
                res = compile_circuit(preset_name, qasm_text, DEFAULT_ARCH_JSON)
                st.session_state["debug_info"]    = res["debug"]
                st.session_state["debug_json"]    = json.dumps(res["debug"])
                st.session_state["original_qasm"] = res["original_qasm"]
                st.session_state["native_qasm"]   = res.get("native_qasm", "")
                st.session_state["compiled"]      = True
                st.rerun()

    if not st.session_state.get("compiled"):
        st.title("Quantum Compilation Explorer")
        st.info("Seleziona un circuito e premi Compila.")
        return

    di            = st.session_state["debug_info"]
    dbg_json      = st.session_state["debug_json"]
    original_qasm = st.session_state.get("original_qasm", "")
    native_qasm   = st.session_state.get("native_qasm", "")

    # ── Row 1: title + metrics + circuit diagrams ──────────────────────────
    st.subheader(preset_name)
    m1, m2, _ = st.columns([1, 1, 6])
    m1.metric("Qubits", di["n_qubits"])
    m2.metric("Layers", di["n_layers"])

    with st.expander("Circuiti", expanded=True):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("Originale")
            circ_img = render_circuit_image(original_qasm)
            if circ_img:
                st.image(circ_img, use_container_width=True)
            elif original_qasm:
                st.code(original_qasm, language="qasm")
        with cc2:
            st.caption("Transpilato NAQC (H · P · CZ)")
            native_img = render_circuit_image(native_qasm) if native_qasm else None
            if native_img:
                st.image(native_img, use_container_width=True)
            elif native_qasm:
                st.code(native_qasm, language="qasm")
            else:
                st.info("Non disponibile")

    # ── Row 2: full-width canvas animation ────────────────────────────────
    state_json = json.dumps(
        compute_state_evolution(native_qasm, di["n_qubits"])
    )
    render_realtime_canvas(dbg_json, DEFAULT_ARCH_JSON, state_json, height=860)

    # ── Debug toggle ───────────────────────────────────────────────────────
    with st.expander("🐛 Debug: compiler output", expanded=False):
        st.code(_debug_asm(di), language=None)


if __name__ == "__main__":
    main()
