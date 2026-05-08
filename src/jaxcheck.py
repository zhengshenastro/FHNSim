import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Default backend:", jax.default_backend())
print("Devices:", jax.devices())

x = jnp.ones((4096, 4096), dtype=jnp.float64)
y = jnp.ones((4096, 4096), dtype=jnp.float64)

# 触发一次 JIT 编译 + 运行
f = jax.jit(lambda a, b: a @ b)
z = f(x, y)

print("Result device:", z.device)
print("Result shape:", z.shape, "dtype:", z.dtype)