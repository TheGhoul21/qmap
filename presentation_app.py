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
from pathlib import Path
from typing import Any

import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "python"))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Compilation Explorer",
    page_icon="⚛️",
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
    "Linear chain (8q)": {
        "description": "8-qubit linear CZ chain — semplice, buono per iniziare",
        "builder": "linear_chain",
        "n_qubits": 8,
    },
    "GHZ (6q)": {
        "description": "GHZ state preparation — fan-out di CZ da q0",
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
        "description": "Pattern a brickwork — tipico dei circuiti variazionali",
        "builder": "brickwork",
        "n_qubits": 10,
        "n_layers": 3,
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

    if b == "qasm":
        qc = ir.QuantumComputation.from_qasm_str(qasm_text)
        return qc, qc.num_qubits

    raise ValueError(f"Unknown builder: {b}")


# ── compilation (cached by circuit repr + arch) ────────────────────────────────
@st.cache_data(show_spinner=False)
def compile_circuit(preset_name: str, qasm_text: str, arch_json: str) -> dict[str, Any]:
    """Compile and return debug_info dict. Cached by inputs."""
    import mqt.qmap.na.zoned as zoned

    arch = zoned.ZonedNeutralAtomArchitecture.from_json_string(arch_json)
    preset = PRESETS[preset_name]
    qc, _ = _build_circuit(preset_name, preset, qasm_text)
    c = zoned.RoutingAgnosticCompiler(arch)
    c.compile(qc)
    return c.debug_info()


# ── frame rendering (cached per frame) ────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=128)
def render_frame_png(debug_json: str, arch_json: str, frame: int, figsize_w: float, figsize_h: float) -> bytes:
    """Render a single frame to PNG bytes. Caches up to 128 frames."""
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


def _frame_label(frame: int, n_layers: int) -> str:
    """Human-readable label for a frame index."""
    if frame == 2 * n_layers:
        return f"Frame {frame} — Final storage"
    layer = frame // 2
    if frame % 2 == 0:
        return f"Frame {frame} — Layer {layer}: Transitione / 1Q gates"
    return f"Frame {frame} — Layer {layer}: CZ entanglement ⚡"


def _frame_color(frame: int) -> str:
    if frame % 2 == 1:
        return "#c0392b"
    return "#2980b9"


def _build_frame_table(debug: dict) -> list[dict]:
    """Build a summary table of all frames for the overview."""
    n_layers = debug["n_layers"]
    tq = debug["two_qubit_layers"]
    sq = debug.get("single_qubit_layers", [])
    rows = []
    for f in range(2 * n_layers + 1):
        layer = f // 2
        if f == 2 * n_layers:
            rows.append({"Frame": f, "Type": "✅ Final", "Layer": layer, "Gates": "—"})
        elif f % 2 == 1:
            n_cz = len(tq[layer]) if layer < len(tq) else 0
            rows.append({"Frame": f, "Type": "⚡ CZ", "Layer": layer, "Gates": f"{n_cz} CZ"})
        else:
            n_1q = len(sq[layer]) if layer < len(sq) else 0
            n_cz_next = len(tq[layer]) if layer < len(tq) else 0
            tag = f"{n_1q} 1Q" if n_1q else "—"
            rows.append({"Frame": f, "Type": "🔄 Transit", "Layer": layer, "Gates": tag})
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _goto(frame: int) -> None:
    """Canonical way to change the current frame.

    Updates BOTH the logical key ``st.session_state["frame"]`` AND the slider
    widget key ``st.session_state["_frame_slider"]`` so they stay in sync.
    """
    st.session_state["frame"] = frame
    st.session_state["_frame_slider"] = frame


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── session-state defaults ─────────────────────────────────────────────────
    st.session_state.setdefault("frame", 0)
    st.session_state.setdefault("_frame_slider", 0)
    st.session_state.setdefault("compiled", False)
    st.session_state.setdefault("autoplay", False)
    st.session_state.setdefault("autoplay_interval", 1.5)

    # ─────────────────────────────── SIDEBAR ──────────────────────────────────
    with st.sidebar:
        st.title("⚛️ Quantum Compiler\nPresentation")
        st.markdown("---")

        # ── 1. Circuit picker ─────────────────────────────────────────────────
        st.subheader("1. Scegli il circuito")
        preset_name = st.selectbox(
            "Preset",
            list(PRESETS.keys()),
            help="Seleziona un circuito predefinito o scrivi QASM custom",
        )
        preset = PRESETS[preset_name]
        st.caption(f"_{preset['description']}_")

        qasm_text = ""
        if preset["builder"] == "qasm":
            qasm_text = st.text_area(
                "OpenQASM 3 source",
                value=("OPENQASM 3.0;\nqubit[6] q;\nh q[0];\n"
                       "cz q[0], q[1];\ncz q[1], q[2];\ncz q[2], q[3];\n"
                       "cz q[3], q[4];\ncz q[4], q[5];\nh q[2];\nh q[4];\n"),
                height=220,
            )

        st.markdown("---")

        # ── 2. Compile ────────────────────────────────────────────────────────
        st.subheader("2. Compila")
        if st.button("🚀 Compila ora", type="primary", use_container_width=True):
            st.session_state["compiled"] = False
            _goto(0)
            st.session_state["autoplay"] = False
            with st.spinner("Compilazione in corso…"):
                try:
                    di = compile_circuit(preset_name, qasm_text, DEFAULT_ARCH_JSON)
                    st.session_state["debug_info"] = di
                    st.session_state["debug_json"] = json.dumps(di)
                    st.session_state["compiled"] = True
                    st.session_state["n_layers"] = di["n_layers"]
                    st.session_state["total_frames"] = 2 * di["n_layers"] + 1
                    st.success(
                        f"✅ Compilato!  {di['n_layers']} layer, {di['n_qubits']} qubit"
                    )
                except Exception as exc:
                    st.error(f"Errore:\n```\n{exc}\n```")

        st.markdown("---")

        # ── 3-6. Navigation (only when compiled) ──────────────────────────────
        if st.session_state["compiled"]:
            n_layers: int = st.session_state["n_layers"]
            total: int = st.session_state["total_frames"]
            cur: int = st.session_state["frame"]

            # ── Slider ────────────────────────────────────────────────────────
            # key="_frame_slider" so Streamlit writes user drag to that key.
            # _goto() keeps "frame" and "_frame_slider" in sync.
            st.subheader("3. Naviga i frame")
            slider_val: int = st.slider(
                "Frame",
                min_value=0,
                max_value=total - 1,
                value=cur,
                key="_frame_slider",
                help="Trascina per cambiare frame",
            )
            if slider_val != cur:
                # User dragged the slider → sync the logical key
                st.session_state["frame"] = slider_val
                st.rerun()

            # ── Layer-jump buttons ─────────────────────────────────────────
            st.caption("Vai direttamente a un layer:")
            cols_per_row = 5
            for row_start in range(0, n_layers, cols_per_row):
                row_layers = list(range(row_start, min(row_start + cols_per_row, n_layers)))
                btn_cols = st.columns(len(row_layers))
                for col, li in zip(btn_cols, row_layers):
                    with col:
                        st.button(f"L{li}", key=f"goto_L{li}",
                                  use_container_width=True,
                                  on_click=_goto, args=(li * 2,))

            st.markdown("---")

            # ── Prev / Next ────────────────────────────────────────────────
            st.subheader("4. Controllo frame")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.button("⏮ Prev", use_container_width=True,
                          disabled=(cur == 0),
                          on_click=_goto, args=(cur - 1,))
            with c2:
                st.button("⏭ Next", use_container_width=True,
                          disabled=(cur >= total - 1),
                          on_click=_goto, args=(cur + 1,))
            with c3:
                st.button("⏪ Reset", use_container_width=True,
                          on_click=_goto, args=(0,))

            c4, c5 = st.columns(2)
            with c4:
                st.button("← Layer", use_container_width=True,
                          disabled=(cur < 2),
                          on_click=_goto, args=(cur - 2,))
            with c5:
                st.button("Layer →", use_container_width=True,
                          disabled=(cur >= total - 2),
                          on_click=_goto, args=(cur + 2,))

            st.markdown("---")

            # ── Auto-play ─────────────────────────────────────────────────
            st.subheader("5. Auto-play")
            st.session_state["autoplay_interval"] = st.slider(
                "Intervallo (sec)", 0.5, 5.0,
                float(st.session_state["autoplay_interval"]), 0.1,
                key="_interval_slider",
            )
            ap1, ap2 = st.columns(2)
            with ap1:
                if st.button("▶ Play", use_container_width=True,
                             disabled=st.session_state["autoplay"]):
                    st.session_state["autoplay"] = True
                    st.rerun()
            with ap2:
                if st.button("⏹ Stop", use_container_width=True,
                             disabled=not st.session_state["autoplay"]):
                    st.session_state["autoplay"] = False
                    st.rerun()

            st.markdown("---")

            # ── Figure size ────────────────────────────────────────────────
            st.subheader("6. Opzioni figura")
            st.slider("Larghezza figura", 10.0, 24.0, 17.0, 0.5, key="_fig_w")
            st.slider("Altezza figura", 7.0, 16.0, 10.0, 0.5, key="_fig_h")

    # ─────────────────────────────── MAIN AREA ────────────────────────────────
    if not st.session_state["compiled"]:
        st.title("⚛️ Quantum Compilation Explorer")
        st.markdown(
            """
            Benvenuto! Questa app ti permette di **esplorare layer per layer**
            la compilazione di circuiti quantistici su architetture a **atomi neutri zonati**.

            ### Come usarla
            1. **Scegli** un circuito dal menu a sinistra (o scrivi il tuo in QASM)
            2. Premi **🚀 Compila ora**
            3. Naviga i frame con i tasti **Prev / Next**, lo **slider**, o i bottoni **L0, L1, …**
            4. Usa **Auto-play** per una presentazione fluida senza toccare nulla

            ### Cosa vedrai
            | Frame pari | Frame dispari |
            |:----------:|:-------------:|
            | Atomi in storage + frecce shuttle + gate 1Q | Atomi in ent. zone + gate CZ ⚡ |

            > Ogni coppia (frame pari + dispari) corrisponde a un **layer** della compilazione.
            """
        )
        st.info("👈 Seleziona un circuito e premi **Compila** per iniziare.")
        return

    # ── Read shared state ──────────────────────────────────────────────────────
    frame: int = st.session_state["frame"]
    n_layers: int = st.session_state["n_layers"]
    total: int = st.session_state["total_frames"]
    di: dict = st.session_state["debug_info"]
    fig_w: float = st.session_state.get("_fig_w", 17.0)
    fig_h: float = st.session_state.get("_fig_h", 10.0)
    autoplay: bool = st.session_state["autoplay"]
    interval: float = st.session_state["autoplay_interval"]

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.title(f"⚛️ {preset_name}")
        st.markdown(
            f"**{_frame_label(frame, n_layers)}** — "
            f"frame **{frame}** / {total - 1} &nbsp;|&nbsp; "
            f"layer **{frame // 2}** / {n_layers - 1}"
        )
    with col_badge:
        st.metric("Tipo", "⚡ CZ" if frame % 2 == 1 else "🔄 Transit")
        st.metric("Layer", f"{frame // 2} / {n_layers - 1}")

    # ── Visualization (wrapped in a fragment for flicker-free auto-play) ──────
    # When autoplay is on, run_every=interval causes the fragment to tick every
    # N seconds as a *partial* rerun — no full-page loading spinner.
    # When autoplay is off, run_every=None means the fragment only reruns on
    # explicit user actions (buttons, slider in sidebar).
    run_every: float | None = interval if autoplay else None

    @st.fragment(run_every=run_every)
    def _viz() -> None:
        # ── Auto-advance ──────────────────────────────────────────────────
        if st.session_state["autoplay"]:
            cur = st.session_state["frame"]
            tot = st.session_state["total_frames"]
            if cur < tot - 1:
                st.session_state["frame"] = cur + 1
            else:
                st.session_state["autoplay"] = False
                st.rerun()  # full rerun to update sidebar Stop button state

        f = st.session_state["frame"]
        nl = st.session_state["n_layers"]
        tot = st.session_state["total_frames"]
        fw = st.session_state.get("_fig_w", 17.0)
        fh = st.session_state.get("_fig_h", 10.0)
        dbg_json = st.session_state["debug_json"]
        debug = st.session_state["debug_info"]

        # ── Render image ──────────────────────────────────────────────────
        with st.spinner(f"Rendering frame {f}…"):
            try:
                png = render_frame_png(dbg_json, DEFAULT_ARCH_JSON, f, fw, fh)
                st.image(png, width="stretch")
            except Exception as exc:
                st.error(f"Errore rendering: {exc}")
                return

        # ── Quick-nav row ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Navigazione rapida:**")
        nav_cols = st.columns(9)
        for i, col in enumerate(nav_cols):
            target = f - 4 + i
            if 0 <= target < tot:
                layer_of = target // 2
                sub = "⚡" if target % 2 == 1 else "🔄"
                btn_label = f"{sub} {target}\nL{layer_of}"
                with col:
                    st.button(
                        btn_label,
                        key=f"qnav_{target}",
                        use_container_width=True,
                        type="primary" if target == f else "secondary",
                        on_click=_goto, args=(target,),
                    )
            else:
                col.write(" ")

        # ── Per-frame detail panel ────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 Dettagli frame corrente", expanded=True):
            import math, collections
            layer = f // 2
            is_ent = f % 2 == 1

            dc1, dc2, dc3 = st.columns(3)

            with dc1:
                st.markdown("**🔢 Info**")
                st.write(f"- Qubit: **{debug['n_qubits']}**")
                st.write(f"- Layer totali: **{nl}**")
                st.write(f"- Frame: **{f}** / {tot - 1}")
                st.write(f"- Layer: **{layer}**")
                tipo = "⚡ CZ (Rydberg)" if is_ent else "🔄 Transit / 1Q"
                st.write(f"- Tipo: **{tipo}**")

            with dc2:
                if is_ent:
                    tq = debug.get("two_qubit_layers", [])
                    pairs = tq[layer] if layer < len(tq) else []
                    st.markdown(f"**⚡ CZ — layer {layer}** ({len(pairs)} coppie)")
                    for p in pairs:
                        st.write(f"  CZ(q{p[0]}, q{p[1]})")
                    reuse = debug.get("reuse_qubits", [])
                    if layer < len(reuse) and reuse[layer]:
                        rq = ", ".join(f"q{q}" for q in reuse[layer])
                        st.info(f"♻️ Riusati: {rq}")
                else:
                    sq = debug.get("single_qubit_layers", [])
                    ops = sq[layer] if layer < len(sq) else []
                    if ops:
                        st.markdown(f"**🔦 Gate 1Q — layer {layer}** ({len(ops)})")
                        for op in ops:
                            params = op.get("params", [])
                            def _fa(p: float) -> str:
                                pi = math.pi
                                for d in [1, 2, 4]:
                                    if abs(p * d / pi - round(p * d / pi)) < 1e-6:
                                        n = round(p * d / pi)
                                        return (f"{n}/{d}π" if d > 1
                                                else (f"{n}π" if n != 1 else "π"))
                                return f"{p:.3f}"
                            gate = op["name"]
                            if params:
                                gate += f"({', '.join(_fa(p) for p in params)})"
                            qbs = ", ".join(f"q{q}" for q in op["qubits"])
                            st.write(f"  {gate} → {qbs}")
                    else:
                        st.info("Nessun gate 1Q in questo layer.")

            with dc3:
                st.markdown("**🚀 Shuttle**")
                routing = debug.get("routing", [])
                if not is_ent:
                    if f > 0 and (f - 1) < len(routing):
                        n_s = sum(len(g) for g in routing[f - 1])
                        if n_s:
                            st.write(f"↓ Stored: **{n_s}** atomi")
                    if f < len(routing):
                        n_l = sum(len(g) for g in routing[f])
                        if n_l:
                            st.write(f"↑ Loading: **{n_l}** atomi")
                    if (f >= len(routing) or
                            sum(len(g) for g in routing[f]) == 0) and f == 0:
                        st.info("Frame iniziale.")

                arch_data = json.loads(DEFAULT_ARCH_JSON)
                op_dur = arch_data.get("operation_duration", {})
                t_cz = op_dur.get("rydberg_gate", 0.36)
                t_1q = op_dur.get("single_qubit_gate", 52.0)
                t_tr = op_dur.get("atom_transfer", 15.0)
                if is_ent:
                    st.metric("⏱ Stimato", f"{t_cz:.2f} µs")
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
                        st.metric("⏱ Stimato", f"{t_est:.1f} µs")

        # ── Overview table ────────────────────────────────────────────────
        with st.expander("📊 Overview tutti i frame", expanded=False):
            import pandas as pd
            rows = _build_frame_table(debug)
            df = pd.DataFrame(rows)

            def _hl(row: Any) -> list[str]:
                if row["Frame"] == f:
                    return ["background-color:#fff3cd;font-weight:bold"] * len(row)
                if row["Type"].startswith("⚡"):
                    return ["background-color:#fde8e8"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df.style.apply(_hl, axis=1),
                use_container_width=True,
                height=300,
            )

        st.caption(
            "💡 Usa **Prev/Next** o i bottoni **L0, L1…** nella sidebar. "
            "**Auto-play** avanza automaticamente senza loading spinner."
        )

    _viz()


if __name__ == "__main__" or True:
    main()
