import jax
jax.config.update("jax_enable_x64", True)
# Use GPU if available, fall back to CPU
try:
    jax.config.update('jax_platform_name', 'gpu')
    _ = jax.devices('gpu')
except RuntimeError:
    jax.config.update('jax_platform_name', 'cpu')
print(jax.local_device_count(), flush=True)
print(jax.devices(), flush=True)
from wrapt import partial
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Core finite-difference operators
# ──────────────────────────────────────────────────────────────────────────────

@partial(jit, static_argnames=('constant_values',))
def Laplace_5(u: jnp.ndarray,constant_values: float=1.0) -> jnp.ndarray:
    """
    5-point Laplacian on the N×N interior grid with Dirichlet BC = 1.

    Takes an N×N array of interior values, pads it to (N+2)×(N+2) with the
    boundary (edge) value constant_values, applies the classical 5-point finite-difference
    stencil

        Δ_h u_{i,j} = (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4 u_{i,j}) / h²

    and returns an N×N array.  Grid spacing h = 1/N.
    """
    N = u.shape[0]
    h = 1.0 / N
    u_pad = jnp.pad(u, 1, mode='constant', constant_values=constant_values)   # (N+2, N+2)
    lap = (u_pad[:-2, 1:-1] + u_pad[2:,  1:-1] +
           u_pad[1:-1, :-2] + u_pad[1:-1,  2:] -
           4.0 * u_pad[1:-1, 1:-1]) / h**2
    return lap


@partial(jit, static_argnames=('constant_values',))
def F_residual(u: jnp.ndarray, constant_values: float=1.0) -> jnp.ndarray:
    """Residual F(u) = Δ_h u − u⁴,  shape (N, N)."""
    return Laplace_5(u, constant_values=constant_values) - u**4


# ──────────────────────────────────────────────────────────────────────────────
# Jacobian  J(u) = dF/du = Δ_h − diag(4u³)
# ──────────────────────────────────────────────────────────────────────────────

@partial(jit, static_argnames=('constant_values',))
def J_matvec(u: jnp.ndarray, v: jnp.ndarray, constant_values: float=1.0) -> jnp.ndarray:
    """
    Matrix-vector product J(u)·v without forming the full Jacobian.

    J(u) = Δ_h − diag(4u³).  The perturbation v is a direction in the
    interior, so boundary perturbations are zero (bc = 0).
    """
    lap_v = Laplace_5(v, constant_values=0.0)
    return lap_v - 4.0 * u**3 * v


@jit
def J_diagonal(u: jnp.ndarray) -> jnp.ndarray:
    """
    Diagonal entries of J(u).
    The diagonal of Δ_h is −4/h², so  J_ii = −4/h² − 4u_i³.
    J is strictly diagonally dominant for all u > 0 (off-diagonal sum = 4/h²).
    """
    N = u.shape[0]
    h = 1.0 / N
    return -4.0 / h**2 - 4.0 * u**3

@jit
def get_jacobian_matrix(u: jnp.ndarray) -> jnp.ndarray:
    """
    Build the full N²×N² Jacobian of F_residual via jax.jacfwd.
    Feasible only for small N (≤ ~100); for large N use J_matvec instead.

    Args:
        u: (N, N) interior solution array.

    Returns:
        J: (N², N²) dense Jacobian matrix.
    """
    N = u.shape[0]

    def F_flat(u_flat: jnp.ndarray) -> jnp.ndarray:
        return F_residual(u_flat.reshape(N, N)).reshape(-1)

    return jax.jacfwd(F_flat)(u.reshape(-1))


# ──────────────────────────────────────────────────────────────────────────────
# Error metrics
# ──────────────────────────────────────────────────────────────────────────────

@jit
def get_L2_error(approx: jnp.ndarray, exact: jnp.ndarray) -> jnp.ndarray:
    """L2 norm:  sqrt( mean( (approx − exact)² ) )"""
    return jnp.sqrt(jnp.mean((approx - exact) ** 2))


@jit
def get_inf_error(approx: jnp.ndarray, exact: jnp.ndarray) -> jnp.ndarray:
    """Infinity norm:  max |approx − exact|"""
    return jnp.max(jnp.abs(approx - exact))


# ──────────────────────────────────────────────────────────────────────────────
# Jacobi linear solver  (matrix-free)
# ──────────────────────────────────────────────────────────────────────────────

def _jacobi_solve(
    u: jnp.ndarray,
    rhs: jnp.ndarray,
    tol: float,
    max_iter: int,
) -> jnp.ndarray:
    """
    Solve J(u)·x = rhs using Jacobi iteration.

    Does not form J explicitly; uses J_matvec and J_diagonal.
    J is strictly diagonally dominant for u > 0, guaranteeing convergence.

    Jacobi update:
        x^{k+1} = D^{-1} (rhs − (J − D) x^k)
                = D^{-1} (rhs − J·x^k + D·x^k)

    Returns x of shape (N, N).
    """
    d = J_diagonal(u)        # (N, N)
    d_inv = 1.0 / d

    @jit
    def step(x, u, d_inv, d, rhs):
        Jx = J_matvec(u, x)
        return d_inv * (rhs - Jx + d * x)

    x = d_inv * rhs   # warm start: much better than x=0 for large N
    for _ in range(max_iter):
        x_new = step(x, u, d_inv, d, rhs)
        err = float(jnp.max(jnp.abs(x_new - x)))
        if err < tol:
            return x_new
        x = x_new

    print(f"  Jacobi did NOT converge after {max_iter} iterations (err={err:.2e}).",
          flush=True)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# Problem class
# ──────────────────────────────────────────────────────────────────────────────

class PS4_Problem1:
    """
    Solver for the elliptic BVP

        ∇²u − u⁴ = 0   on  Ω = [0,1]²
        u = 1           on  ∂Ω

    Discretised on an N×N interior grid (N interior points per dimension).
    Grid spacing h = 1/(N+1); full grid including boundary is (N+2)×(N+2).

    Args:
        N: Number of interior grid points per dimension.
    """

    def __init__(self, N: int = 64):
        assert N > 1, "N must be greater than 1."
        self.N = N
        self.h = 1.0 / (N + 1)
        # Interior coordinates:  h, 2h, …, Nh  ∈ (0, 1)
        self.x_coordinate = jnp.linspace(0.0, 1.0, N + 2)[1:-1]   # (N,)
        self.y_coordinate = self.x_coordinate
        self.X, self.Y = jnp.meshgrid(self.x_coordinate, self.y_coordinate,
                                       indexing='ij')
        self.u0 = jnp.ones((N, N), dtype=jnp.float64)   # initial guess = bc = 1
        self.u_direct = None
        self.u_jacobi = None
        print(f"Initialized PS4_Problem1 with N={N}, h={self.h:.6f}", flush=True)

    # ------------------------------------------------------------------
    # (b) Newton with direct solve of J·Δu = F
    # ------------------------------------------------------------------

    def solve_direct(self, tol: float = 1e-10, max_iter: int = 50) -> jnp.ndarray:
        """
        Newton-Raphson: solve J(u)·Δu = F(u) via jnp.linalg.solve each step.
        Builds the full N²×N² Jacobian with jax.jacfwd (dense, LU factorisation).
        Suitable for small N (≤ ~100).

        Convergence criterion: ‖Δu‖_∞ < tol.
        """
        u = self.u0
        for k in range(max_iter):
            f_vec  = F_residual(u).reshape(-1)              # (N²,)
            J_mat  = get_jacobian_matrix(u)                  # (N², N²)
            delta_u = jnp.linalg.solve(J_mat, f_vec)        # (N²,)
            u = u - delta_u.reshape(self.N, self.N)
            res = float(jnp.max(jnp.abs(delta_u)))
            print(f"  Newton iter {k+1}: ‖Δu‖_∞ = {res:.3e}", flush=True)
            if res < tol:
                print(f"Newton (direct) converged in {k+1} iterations.", flush=True)
                break
        else:
            print("Newton (direct) did NOT converge.", flush=True)
        self.u_direct = u
        return u

    # ------------------------------------------------------------------
    # (c) Newton with Jacobi inner solver (matrix-free)
    # ------------------------------------------------------------------

    def solve_jacobi(
        self,
        newton_tol: float = 1e-5,
        jacobi_tol: float = None,
        max_newton: int = 100,
        max_jacobi: int = None,
    ) -> jnp.ndarray:
        """
        Newton-Raphson: each linear step J(u)·Δu = F(u) solved by Jacobi.
        No full matrix is formed; uses J_matvec and J_diagonal.
        Suitable for large N.

        Newton convergence criterion:  ‖Δu‖_∞ < newton_tol.
        Inner Jacobi criterion:        ‖x^{k+1} − x^k‖_∞ < jacobi_tol.

        Note: Newton can only converge to ~jacobi_tol accuracy (inexact Newton).
        Keep newton_tol ≥ jacobi_tol, or pass a tighter jacobi_tol explicitly.
        """
        if jacobi_tol is None:
            jacobi_tol = 0.19 / self.N**2
        if max_jacobi is None:
            max_jacobi = int(self.N * 100)

        u = self.u0
        for k in range(max_newton):
            f = F_residual(u)                                         # (N, N)
            delta_u = _jacobi_solve(u, f, tol=jacobi_tol, max_iter=max_jacobi)
            u = u - delta_u
            res = float(jnp.max(jnp.abs(delta_u)))
            print(f"  Newton iter {k+1}: ‖Δu‖_∞ = {res:.3e}", flush=True)
            if res < newton_tol:
                print(f"Newton (Jacobi) converged in {k+1} iterations.", flush=True)
                break
        else:
            print("Newton (Jacobi) did NOT converge.", flush=True)
        self.u_jacobi = u
        return u

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, u: jnp.ndarray, name: str = 'u', max_N: int = 512):
        """Two-panel: 2D colormap (left) + 1D slice along y ≈ 0.5 (right)."""
        step = max(1, self.N // max_N)
        u_np  = jax.device_get(u)
        x_np  = jax.device_get(self.x_coordinate)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        pcm = ax1.pcolormesh(
            self.X[::step, ::step], self.Y[::step, ::step],
            u_np[::step, ::step], shading='auto', cmap='hot'
        )
        fig.colorbar(pcm, ax=ax1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        ax1.set_title(name)
        ax2.plot(x_np, u_np[:, self.N // 2])
        ax2.set_xlabel('x')
        ax2.set_ylabel(name)
        ax2.set_title(f'{name} along y = {float(self.y_coordinate[self.N // 2]):.3f}')
        plt.tight_layout()
        plt.show()
