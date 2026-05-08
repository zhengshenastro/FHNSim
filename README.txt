
# This project is modified from https://github.com/RJ-9H/FHN-Projecy/tree/main
# Accelerated with JAX

This project supports three execution backends:

1. **NumPy (baseline CPU implementation)**
2. **JAX on CPU**
3. **JAX on CUDA GPU (recommended)**

---

# 1. JAX on CUDA (Recommended)

The GPU backend provides the largest performance improvement and is strongly recommended for large-scale simulations.

## Expected Performance

On our benchmark system:

- **CPU:** Intel i9-13950HX
- **GPU:** NVIDIA RTX 4000 Ada

the CUDA backend achieves approximately:

```text
~10× speedup