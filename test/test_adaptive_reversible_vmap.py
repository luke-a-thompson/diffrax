import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

def test_adaptive_reversible_vmap_unequal_step_counts():
    adjoint = diffrax.ReversibleAdjoint()
    keys = jax.random.split(jax.random.key(21), 2)
    y0s = jnp.array([.1, 2.])
    def solve(parameter, y0, key, adjoint):
        brownian = diffrax.VirtualBrownianTree(0., 1., tol=1e-5, shape=(), key=key)
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(lambda t, y, p: -p*y**3),
            diffrax.ControlTerm(lambda t, y, p: jnp.asarray(.2), brownian),
        )
        return diffrax.diffeqsolve(
            terms, diffrax.ReversibleHeun(), t0=0., t1=1., dt0=.2,
            y0=y0, args=parameter, adjoint=adjoint,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3, dtmin=1e-5),
            max_steps=512, saveat=diffrax.SaveAt(ts=jnp.linspace(0., 1., 11)),
        )
    parameter = jnp.asarray(.5)
    counts = eqx.filter_jit(lambda: jax.vmap(
        lambda y, k: solve(parameter, y, k, adjoint).stats['num_accepted_steps'])(y0s, keys))()
    assert counts[0] != counts[1]
    @eqx.filter_jit
    def value_and_grad(adjoint):
        def loss(parameter):
            paths = jax.vmap(lambda y, k: solve(parameter, y, k, adjoint).ys)(y0s, keys)
            return jnp.mean(paths**2)
        return jax.value_and_grad(loss)(parameter)
    reference = value_and_grad(diffrax.RecursiveCheckpointAdjoint())
    actual = value_and_grad(adjoint)
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=1e-6)
