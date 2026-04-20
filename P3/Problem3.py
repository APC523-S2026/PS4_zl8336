import jax
jax.config.update("jax_enable_x64", True)
try:
    jax.config.update('jax_platform_name', 'gpu')
    _ = jax.devices('gpu')
    print("Using GPU:", jax.devices('gpu'), flush=True)
except RuntimeError:
    jax.config.update('jax_platform_name', 'cpu')
    print("GPU not available, using CPU.", flush=True)
print(jax.local_device_count(), flush=True)
print(jax.devices(), flush=True)

import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────

A     = 1.0        # advection speed in x
B     = 2.0        # advection speed in y
SIGMA = 3.0 / 20.0 # Gaussian half-width σ


# ──────────────────────────────────────────────────────────────────────────────
# Module-level functions
# ──────────────────────────────────────────────────────────────────────────────

@jit
def initial_condition(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """Gaussian IC: exp(-((x-0.5)^2 + (y-0.5)^2) / σ^2)."""
    return jnp.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / SIGMA**2)


@jit
def analytical_solution(
    X: jnp.ndarray, 
    Y: jnp.ndarray,
    t, a, b
    ) -> jnp.ndarray:
    """
    Exact solution: translate IC with periodic wrapping.
        u(t, x, y) = u(0, (x − at) mod 1, (y − bt) mod 1)
    """
    x_mod = (X - a * t) % 1.0
    y_mod = (Y - b * t) % 1.0
    return jnp.exp(-((x_mod - 0.5)**2 + (y_mod - 0.5)**2) / SIGMA**2)


@jit
def ctu_step(u: jnp.ndarray, mu: float, nu: float) -> jnp.ndarray:
    """
    Corner Transport Upstream (CTU) single step.

    Exact solution of piecewise-constant IC projected back to the initial mesh:
        u^{n+1}_{i,j} = (1−μ)(1−ν) u_{i,j}
                       +  μ (1−ν) u_{i−1,j}
                       + (1−μ) ν  u_{i,j−1}
                       +  μ ν     u_{i−1,j−1}

    μ = a·dt/dx  (x Courant number),  ν = b·dt/dy  (y Courant number).
    jnp.roll(u, +1, axis=0) → u[i−1, j]  (periodic BC automatic).
    """
    # jnp.roll is used instead of slices like u[:-1, :] because the domain is
    # periodic: cell i=0 must see i=N-1 as its upstream neighbour.  roll wraps
    # cyclically (new[0,:] = old[N-1,:]), whereas u[:-1,:] drops the last row
    # and loses the wrap-around entirely.
    u_xm   = jnp.roll(u,    1, axis=0)   # u[i−1, j]
    u_ym   = jnp.roll(u,    1, axis=1)   # u[i, j−1]
    u_xmym = jnp.roll(u_xm, 1, axis=1)  # u[i−1, j−1]
    return ((1 - mu) * (1 - nu) * u
            + mu * (1 - nu) * u_xm
            + (1 - mu) * nu * u_ym
            + mu * nu       * u_xmym)


@jit
def lax_wendroff_step(u: jnp.ndarray, mu: float, nu: float) -> jnp.ndarray:
    """
    Lax-Wendroff single step for 2D constant-coefficient advection.

    Derived from second-order Taylor expansion in time, replaced by spatial
    derivatives via the PDE:
        u^{n+1} = u^n
                − μ/2   (u_{i+1,j} − u_{i−1,j})
                − ν/2   (u_{i,j+1} − u_{i,j−1})
                + μ²/2  (u_{i+1,j} − 2u_{i,j} + u_{i−1,j})
                + ν²/2  (u_{i,j+1} − 2u_{i,j} + u_{i,j−1})
                + μν/4  (u_{i+1,j+1} − u_{i+1,j−1} − u_{i−1,j+1} + u_{i−1,j−1})

    μ = a·dt/dx,  ν = b·dt/dy.
    """
    u_xp = jnp.roll(u, -1, axis=0)   # u[i+1, j]
    u_xm = jnp.roll(u,  1, axis=0)   # u[i−1, j]
    u_yp = jnp.roll(u, -1, axis=1)   # u[i, j+1]
    u_ym = jnp.roll(u,  1, axis=1)   # u[i, j−1]
    u_xpyp = jnp.roll(u_xp, -1, axis=1)  # u[i+1, j+1]
    u_xpym = jnp.roll(u_xp,  1, axis=1)  # u[i+1, j−1]
    u_xmyp = jnp.roll(u_xm, -1, axis=1)  # u[i−1, j+1]
    u_xmym = jnp.roll(u_xm,  1, axis=1)  # u[i−1, j−1]
    return (u
            - mu / 2    * (u_xp - u_xm)
            - nu / 2    * (u_yp - u_ym)
            + mu**2 / 2 * (u_xp - 2*u + u_xm)
            + nu**2 / 2 * (u_yp - 2*u + u_ym)
            + mu * nu / 4 * (u_xpyp - u_xpym - u_xmyp + u_xmym))


@jit
def get_L2_error(approx: jnp.ndarray, exact: jnp.ndarray) -> jnp.ndarray:
    """RMS error: sqrt(mean((approx − exact)^2))."""
    return jnp.sqrt(jnp.mean((approx - exact)**2))


# ──────────────────────────────────────────────────────────────────────────────
# Problem class
# ──────────────────────────────────────────────────────────────────────────────

class PS4_Problem3:
    """
    Solver for 2D advection  ∂u/∂t + a∂u/∂x + b∂u/∂y = 0
    on [0,1]² with periodic BC and Gaussian IC.

    Time loop uses jax.lax.scan (single XLA compilation, GPU-friendly).
    Two methods supported: CTU (1st-order) and Lax-Wendroff (2nd-order).

    Args:
        N: number of cells per dimension.
        a, b: advection speeds.
        C: CFL number; dt = C * dx / sqrt(a² + b²).
    """

    def __init__(self, N: int = 128, a: float = A, b: float = B, C: float = 0.5):
        assert N > 0, "N must be positive."
        self.N  = N
        self.a  = float(a)
        self.b  = float(b)
        dx = dy = 1.0 / N
        self.dx = dx
        self.dy = dy
        dt = C * dx / float(np.sqrt(a**2 + b**2))
        self.dt = dt
        self.mu = a * dt / dx   # x Courant number
        self.nu = b * dt / dy   # y Courant number
        # Cell-centre coordinates
        x_coord = jnp.linspace(0.0, 1.0, N, endpoint=False) + dx / 2
        y_coord = jnp.linspace(0.0, 1.0, N, endpoint=False) + dy / 2
        self.X, self.Y = jnp.meshgrid(x_coord, y_coord, indexing='ij')
        self.u0 = initial_condition(self.X, self.Y)
        print(f"Initialized PS4_Problem3: N={N}, dt={dt:.5e}, "
              f"μ={self.mu:.4f}, ν={self.nu:.4f}", flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Internal: jax.lax.scan runner
    # ──────────────────────────────────────────────────────────────────

    def _run_scan(self, u0: jnp.ndarray, step_fn, N_steps: int,
                  t_offset: float):
        """
        Advance u0 by N_steps steps with jax.lax.scan.

        Carry: (u, t).  Output per step: scalar L2 error vs exact solution.
        t_offset is the physical time at the start of this scan segment.

        Returns:
            u_final: (N, N) solution after N_steps steps.
            l2_all:  (N_steps,) L2 error at each step.
        """
        X, Y     = self.X, self.Y
        a, b, dt = self.a, self.b, self.dt

        def scan_fn(carry, _):
            u, t   = carry
            u_new  = step_fn(u)
            t_new  = t + dt
            l2     = get_L2_error(u_new, analytical_solution(X, Y, t_new, a, b))
            return (u_new, t_new), l2

        (u_final, _), l2_all = jax.lax.scan(
            scan_fn,
            (u0, jnp.array(t_offset, dtype=jnp.float64)),
            None,
            length=N_steps,
        )
        return u_final, l2_all

    # ──────────────────────────────────────────────────────────────────
    # Public: full simulation
    # ──────────────────────────────────────────────────────────────────

    def solve(self, t_stop: float = 10.0, method: str = 'CTU') -> dict:
        """
        Run simulation from t=0 to t=t_stop using jax.lax.scan.

        Runs in two scan stages (0→1 and 1→t_stop) to capture a snapshot at
        t=1 without storing the full trajectory.

        Args:
            t_stop: end time (default 10).
            method: 'CTU' or 'LW'.

        Returns dict with keys:
            u_at_1     (N, N)  solution at t=1
            u_at_final (N, N)  solution at t=t_stop
            t_all      (N_total,) time at each step
            l2_all     (N_total,) L2 error at each step
            method, N
        """
        mu, nu = self.mu, self.nu
        if method == 'CTU':
            step_fn = lambda u: ctu_step(u, mu, nu)
        elif method == 'LW':
            step_fn = lambda u: lax_wendroff_step(u, mu, nu)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'CTU' or 'LW'.")

        n1      = int(round(1.0 / self.dt))
        n_total = int(round(t_stop / self.dt))
        n_rest  = n_total - n1

        print(f"[{method}] N={self.N}, N_total={n_total}, dt={self.dt:.3e}, "
              f"μ={mu:.4f}, ν={nu:.4f}", flush=True)

        # Stage 1: t = 0 → 1
        u_at_1, l2_0to1 = self._run_scan(self.u0, step_fn, n1, t_offset=0.0)
        print(f"  t = 0 → 1       done,  L2(t=1)       = {float(l2_0to1[-1]):.3e}",
              flush=True)

        # Stage 2: t = 1 → t_stop  (skipped when t_stop == 1)
        if n_rest > 0:
            u_at_final, l2_1to_final = self._run_scan(u_at_1, step_fn, n_rest,
                                                       t_offset=1.0)
            print(f"  t = 1 → {t_stop}  done,  L2(t={t_stop}) = {float(l2_1to_final[-1]):.3e}",
                  flush=True)
            t_all  = jnp.concatenate([
                jnp.arange(1, n1 + 1,           dtype=jnp.float64) * self.dt,
                1.0 + jnp.arange(1, n_rest + 1, dtype=jnp.float64) * self.dt,
            ])
            l2_all = jnp.concatenate([l2_0to1, l2_1to_final])
        else:
            u_at_final = u_at_1
            t_all      = jnp.arange(1, n1 + 1, dtype=jnp.float64) * self.dt
            l2_all     = l2_0to1

        return {
            'u_at_1':     u_at_1,
            'u_at_final': u_at_final,
            't_all':      t_all,
            'l2_all':     l2_all,
            'method':     method,
            'N':          self.N,
        }


    # ──────────────────────────────────────────────────────────────────
    # Snapshots at integer times
    # ──────────────────────────────────────────────────────────────────

    def get_integer_snapshots(self, t_stop: float = 10.0,
                               method: str = 'CTU') -> list:
        """
        Run the simulation in unit segments, returning the field at each
        integer time t = 0, 1, 2, …, int(t_stop).

        Each segment is a separate jax.lax.scan call that re-uses the same
        compiled kernel, so there is no recompilation overhead after the first.

        Returns:
            snapshots: list of (N, N) arrays, len = int(t_stop) + 1.
        """
        mu, nu = self.mu, self.nu
        if method == 'CTU':
            step_fn = lambda u: ctu_step(u, mu, nu)
        elif method == 'LW':
            step_fn = lambda u: lax_wendroff_step(u, mu, nu)
        else:
            raise ValueError(f"Unknown method '{method}'.")

        n_per_unit = int(round(1.0 / self.dt))
        n_units    = int(round(t_stop))

        snapshots = [self.u0]
        u = self.u0
        for k in range(n_units):
            u, _ = self._run_scan(u, step_fn, n_per_unit, t_offset=float(k))
            snapshots.append(u)
            print(f"  [{method}] snapshot t={k+1}", flush=True)
        return snapshots

    # ──────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────

    def plot(self, result: dict, t_final: float = 10.0):
        """
        Three-panel figure:
          left:   u(x,y) contour at t=1
          centre: u(x,y) contour at t=t_final
          right:  L2 error vs time (all steps + integer-time markers)
        """
        X_np   = jax.device_get(self.X)
        Y_np   = jax.device_get(self.Y)
        u1_np  = jax.device_get(result['u_at_1'])
        uf_np  = jax.device_get(result['u_at_final'])
        t_np   = jax.device_get(result['t_all'])
        l2_np  = jax.device_get(result['l2_all'])
        method = result['method']
        N      = result['N']

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{method},  N={N}', fontsize=16)

        for ax, u_np, tlabel in [(ax1, u1_np, 't = 1'),
                                  (ax2, uf_np, f't = {t_final}')]:
            cf = ax.contourf(X_np, Y_np, u_np, levels=20, cmap='viridis')
            fig.colorbar(cf, ax=ax)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_aspect('equal')
            ax.set_title(f'u(x,y) at {tlabel}', fontsize=12)

        # L2 at integer times
        t_int  = np.arange(1, int(round(t_final)) + 1, dtype=float)
        l2_int = np.array([float(l2_np[np.argmin(np.abs(t_np - ti))])
                           for ti in t_int])
        ax3.plot(t_np, l2_np, alpha=0.3, lw=0.8, color='tab:blue')
        ax3.plot(t_int, l2_int, 'o-', ms=5, color='tab:blue')
        ax3.set_xlabel('t', fontsize=12)
        ax3.set_ylabel('$L_2$ error', fontsize=12)
        ax3.set_yscale('log')
        ax3.set_title('$L_2$ error vs time', fontsize=12)
        ax3.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.show()
