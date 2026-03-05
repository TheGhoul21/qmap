---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Visualizing Zoned Neutral Atom Compilation

MQT QMAP provides a set of visualization utilities that let you inspect every
intermediate step of the zoned neutral atom compiler pipeline:

- **Architecture layout** — storage zones, entanglement zones, SLM trap arrays,
  and Rydberg laser ranges drawn to scale.
- **Compilation step-by-step** — per-layer views of where each atom sits, which
  atoms move between layers (with curved arrows), and which CZ gates fire in the
  entanglement zone.
- **Full animation** — an animated GIF / MP4 / interactive HTML widget that
  plays through every transition and gate layer automatically.

All functions live in `mqt.qmap.visualization` and require
[`matplotlib`](https://matplotlib.org/) (`pip install mqt.qmap[visualization]`).

## Architecture Overview

```{code-cell} ipython3
import json
from mqt.qmap.visualization import visualize_architecture

ARCH_JSON = json.dumps({
    "name": "compact_32q_architecture",
    "operation_duration": {"rydberg_gate": 0.36, "single_qubit_gate": 52, "atom_transfer": 15},
    "operation_fidelity": {"rydberg_gate": 0.995, "single_qubit_gate": 0.9997, "atom_transfer": 0.999},
    "qubit_spec": {"T": 1.5e6},
    "storage_zones": [{
        "zone_id": 0,
        "slms": [{"id": 0, "site_separation": [6, 6], "r": 4, "c": 8, "location": [0, 0]}],
        "offset": [0, 0], "dimension": [48, 24]
    }],
    "entanglement_zones": [{
        "zone_id": 0,
        "slms": [
            {"id": 1, "site_separation": [6, 6], "r": 2, "c": 16, "location": [1, 38]},
            {"id": 2, "site_separation": [6, 6], "r": 2, "c": 16, "location": [3, 38]}
        ],
        "offset": [1, 38], "dimension": [98, 14]
    }],
    "aods": [{"id": 0, "site_separation": 2, "r": 8, "c": 16}],
    "arch_range": [[-5, -5], [105, 60]],
    "rydberg_range": [[[-5, 33], [105, 60]]]
})

fig = visualize_architecture(ARCH_JSON)
```

`visualize_architecture` draws all SLM trap sites as dots, highlights storage
zones (blue) and entanglement zones (orange), and overlays the Rydberg laser
range (purple dashed rectangle).  The architecture above has:

- A **4 × 8 storage SLM** (32 trap sites) at the top.
- **Two 2 × 16 entanglement SLMs** directly below, offset by 2 µm in *x* so
  that column-aligned pairs across the two SLMs are within Rydberg interaction
  range (the physical requirement for CZ gates).

## Compiling a Circuit

We use a QFT-8 circuit to keep the example concise.  Every two-qubit gate must
be a CZ gate because that is the native two-qubit gate of the architecture.

```{code-cell} ipython3
import mqt.core.ir as ir
from mqt.qmap.na.zoned import ZonedNeutralAtomArchitecture, RoutingAgnosticCompiler

def build_qft(n: int) -> ir.QuantumComputation:
    """Build an n-qubit QFT circuit using H and CZ gates."""
    qc = ir.QuantumComputation(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cz(i, j)
    return qc

arch = ZonedNeutralAtomArchitecture.from_json_string(ARCH_JSON)
qc = build_qft(8)

compiler = RoutingAgnosticCompiler(arch)
compiler.compile(qc)
debug = compiler.debug_info()

print(f"Qubits : {debug['n_qubits']}")
print(f"Layers : {debug['n_layers']}")
print(f"Frames : {2 * debug['n_layers'] + 1}")
```

`debug_info()` returns a dictionary with all intermediate compiler data:

| Key | Description |
|-----|-------------|
| `n_qubits` | Number of qubits |
| `n_layers` | Number of two-qubit gate layers |
| `two_qubit_layers` | `[layer][pair] = [q0, q1]` |
| `single_qubit_layers` | `[layer][op] = {name, qubits}` |
| `placement` | `[layer][qubit] = [slm_id, row, col]` |
| `routing` | `[transition][group] = [qubits]` |
| `reuse_qubits` | `[transition] = [qubits]` (qubits that skip the return to storage) |

## Inspecting Individual Frames

`visualize_compilation_step` renders a single frame.  Frames alternate between
two views:

- **Even frames** (0, 2, 4, …) — atoms in their storage positions.  Curved
  arrows show which atoms will move to the entanglement zone for the next layer,
  colour-coded by qubit index.  Small qubit labels sit at the midpoint of each
  arrow.
- **Odd frames** (1, 3, 5, …) — atoms in the entanglement zone.  Active CZ
  pairs are connected by a red line with a star at the midpoint.  Qubits that
  are *reused* (they stay in the entanglement zone rather than returning to
  storage) are highlighted with a green circle.

```{code-cell} ipython3
from mqt.qmap.visualization import visualize_compilation_step

# Frame 0: initial placement + movement arrows for layer 0
fig = visualize_compilation_step(debug, ARCH_JSON, frame=0)
```

```{code-cell} ipython3
# Frame 1: atoms in the entanglement zone executing CZ gates in layer 0
fig = visualize_compilation_step(debug, ARCH_JSON, frame=1)
```

## Resource Guard

Before compiling large circuits, call `compilation_guard` to get an upfront
estimate of resource usage and animation size:

```{code-cell} ipython3
from mqt.qmap.visualization.visualize_na_compilation import compilation_guard

report = compilation_guard(qc, compiler="agnostic")
print(report)
```

The guard warns if:

- `RoutingAwareCompiler` `max_nodes` would require more than ~4 GB of RAM.
- The circuit has more than 24 qubits with the routing-aware compiler.
- The estimated animation frame count exceeds `max_animation_frames` (default 200).

Pass `raise_on_danger=True` to turn warnings into exceptions for use in
automated pipelines.

## Animating the Full Compilation

`animate_compilation` produces a `matplotlib.animation.FuncAnimation` object
that plays through all frames automatically.

```{code-cell} ipython3
:tags: [remove-output]
from mqt.qmap.visualization import animate_compilation
from IPython.display import HTML

anim = animate_compilation(debug, ARCH_JSON, interval=700)

# Display inline in a Jupyter notebook:
HTML(anim.to_jshtml())
```

To save the animation as a file:

```python
anim.save("qft8.gif", writer="pillow", fps=1.5)   # needs Pillow
anim.save("qft8.mp4", fps=2)                       # needs ffmpeg
```

The `max_frames` parameter limits the animation to the first *N* frames, which
is useful for quick previews of long compilations:

```python
anim = animate_compilation(debug, ARCH_JSON, interval=500, max_frames=20)
```

## Larger Circuits

The same workflow applies to any circuit size.  The `RoutingAgnosticCompiler`
scales to hundreds of qubits in seconds; the `RoutingAwareCompiler` gives better
routing at the cost of an A* search (use `max_nodes` to bound memory).

```{code-cell} ipython3
import time

for n in [8, 16, 32]:
    qc_n = build_qft(n)
    c = RoutingAgnosticCompiler(arch)
    t0 = time.perf_counter()
    c.compile(qc_n)
    dt = time.perf_counter() - t0
    d = c.debug_info()
    print(f"QFT-{n:2d}: {d['n_layers']:3d} layers  {dt*1000:.1f} ms")
```

## Shor's Algorithm — Quantum Phase Estimation

Shor's algorithm finds the order of a modular exponentiation function using
**Quantum Phase Estimation (QPE)**.  The circuit has two registers:

- **Counting register** (*n* qubits) — prepared in uniform superposition by
  Hadamard gates, then measured to extract the period.
- **Work register** (*m* qubits) — holds the eigenstate of the oracle unitary
  U (the modular-exponentiation map).

The gate structure is:

1. H on each counting qubit.
2. For each counting qubit *k*: controlled-U^(2^k) on the work register
   (CZ bonds between qubit *k* and each work qubit, plus entanglement within
   the work register to model arithmetic).
3. Inverse QFT on the counting register (CZ ladder + H gates).

```{code-cell} ipython3
def build_shor_qpe(n_count: int, n_work: int) -> ir.QuantumComputation:
    """QPE circuit for Shor's order-finding algorithm (CZ-basis).

    Args:
        n_count: Size of the phase-estimation (counting) register.
        n_work:  Size of the work register (log2 of the modulus N).

    Returns:
        A :class:`~mqt.core.ir.QuantumComputation` in {H, CZ} gate set.
    """
    n = n_count + n_work
    qc = ir.QuantumComputation(n)
    qc.x(n_count)                          # |work⟩ = |1⟩ (eigenstate of U)
    for i in range(n_count):
        qc.h(i)                            # counting register → |+⟩^n
    for k in range(n_count):              # controlled-U^(2^k) on work register
        for j in range(n_work):
            qc.cz(k, n_count + j)
        for j in range(n_work - 1):       # intra-register arithmetic entanglement
            qc.cz(n_count + j, n_count + j + 1)
    for i in range(n_count - 1, -1, -1): # inverse QFT on counting register
        for j in range(i):
            qc.cz(j, i)
        qc.h(i)
    return qc
```

The circuit scales cleanly: adding counting qubits increases phase resolution
without blowing up compilation time.

```{code-cell} ipython3
configs = [(4, 4), (8, 4), (12, 4), (16, 4), (20, 4), (24, 4), (28, 4)]
for n_count, n_work in configs:
    qc_s = build_shor_qpe(n_count, n_work)
    n2q = sum(1 for op in qc_s if op.num_controls + op.num_targets == 2)
    c = RoutingAgnosticCompiler(arch)
    t0 = time.perf_counter()
    c.compile(qc_s)
    dt = time.perf_counter() - t0
    d = c.debug_info()
    print(
        f"QPE {n_count}+{n_work}={n_count+n_work:2d}q  "
        f"{n2q:4d} CZ  {d['n_layers']:3d} layers  {dt*1000:.0f} ms"
    )
```

The full 24-qubit QPE (20 counting + 4 work, 330 CZ gates) compiles in ~7 ms.
Here are the first two frames of its compilation:

```{code-cell} ipython3
qc_shor = build_shor_qpe(20, 4)
compiler_shor = RoutingAgnosticCompiler(arch)
compiler_shor.compile(qc_shor)
debug_shor = compiler_shor.debug_info()

# Frame 0: initial placement — yellow rings show single-qubit gates (H, X),
# curved arrows show which atoms move down to the entanglement zone.
fig = visualize_compilation_step(debug_shor, ARCH_JSON, frame=0, figsize=(11, 6))
```

```{code-cell} ipython3
# Frame 1: first CZ layer fires.  The Rydberg laser zone glows red,
# the entanglement zone turns scarlet, and a bond line connects the CZ pair.
fig = visualize_compilation_step(debug_shor, ARCH_JSON, frame=1, figsize=(11, 6))
```

To generate the full animation:

```{code-cell} ipython3
:tags: [remove-output]
anim_shor = animate_compilation(debug_shor, ARCH_JSON, interval=500, max_frames=60)
HTML(anim_shor.to_jshtml())
```

```python
# Save locally
anim_shor.save("shor_24q.gif", writer="pillow", fps=2)
```
