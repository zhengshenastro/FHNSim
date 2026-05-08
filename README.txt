# This project is modified from https://github.com/RJ-9H/FHN-Projecy/tree/main
# With JAX Acceleration Backend

This project supports three execution backends:

1. NumPy (baseline CPU implementation)
2. JAX on CPU
3. JAX on CUDA GPU (recommended)

--------------------------------------------------

# 1. JAX on CUDA (Recommended)

The GPU backend provides the largest performance improvement and is strongly recommended for large-scale simulations.

## Expected Performance

On our benchmark system:

- CPU: Intel i9-13950HX
- GPU: NVIDIA RTX 4000 Ada

the CUDA backend achieves approximately:

~10× speedup

compared to the CPU implementation.

The exact acceleration depends on:

- lattice size
- simulation length
- memory bandwidth
- boundary conditions
- visualization frequency

but GPU acceleration becomes especially significant for large 2D systems.

--------------------------------------------------

# 2. Requirements for JAX on CUDA

JAX GPU support is currently most reliable under:

- native Linux
or
- WSL2 (Windows Subsystem for Linux)

Pure Windows CUDA support for JAX is still limited and generally not recommended.

--------------------------------------------------

## Option A — Native Linux (Recommended)

Recommended environment:

- Ubuntu 22.04 / 24.04
- NVIDIA GPU with CUDA support
- Recent NVIDIA driver
- Python 3.10+

### Required components

You must install:

- NVIDIA driver
- CUDA toolkit / runtime
- cuDNN
- JAX CUDA build

A quick sanity check:

python -c "import jax; print(jax.devices())"

should display a CUDA GPU device.

--------------------------------------------------

## Option B — WSL2 on Windows

If you are using Windows, the recommended setup is:

Windows
→ WSL2 (Ubuntu)
→ CUDA passthrough
→ Python virtual environment
→ JAX CUDA

### Requirements

You need:

- Windows 11 (recommended)
- WSL2 enabled
- Ubuntu installed through WSL
- NVIDIA driver with WSL CUDA support INSTALLED ON WIN NOT ON LINUX

Inside WSL:

Nvidia-smi may not correctly detect the GPU, this doesn't matter

Then install the dependencies EXACTLY according to requirements.txt to a virtual env on wsl.

--------------------------------------------------

# 3. If You Cannot Configure WSL / CUDA

The project can still run entirely on Windows using:

JAX on CPU

This mode does not require:

- CUDA
- WSL2
- Linux

and is significantly easier to configure.

--------------------------------------------------

## Expected CPU Performance

Compared to the original NumPy implementation:

JAX on CPU ≈ 2× faster

in typical workloads.

This acceleration mainly comes from:

- XLA compilation
- vectorization
- optimized array execution

although it is still substantially slower than GPU execution.

--------------------------------------------------

# 4. Recommended Backend Priority

Recommended order:

JAX on CUDA (Linux / WSL2)
    ↓
JAX on CPU
    ↓
Pure NumPy

--------------------------------------------------

# 5. Notes

- The first JAX execution may take longer due to JIT compilation.
- GPU acceleration is most effective for large grids and long simulations.
- Small systems may not fully utilize GPU parallelism.
- Numerical results between NumPy and JAX may differ slightly because of:
  - floating-point precision
  - XLA optimization
  - parallel execution ordering

These differences are expected and usually negligible for qualitative dynamics.
