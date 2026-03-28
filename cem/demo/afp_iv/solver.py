from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from optuna.distributions import FloatDistribution, IntDistribution
from tjax import RngStream, frozendict

from cem.models import AFP
from cem.structure.graph import Model, Node
from cem.structure.graph.input_node import InputNode
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, float_field, int_field

from .problem import IVProblem


class AFPIVSolver(Solver[IVProblem]):
    alpha: float = float_field(default=1.5, domain=FloatDistribution(0.1, 4.0), optimize=True)
    beta: float = float_field(default=1.2, domain=FloatDistribution(0.1, 4.0), optimize=True)
    gamma: float = float_field(default=2.0, domain=FloatDistribution(0.1, 4.0), optimize=True)
    delta: float = float_field(default=1.0, domain=FloatDistribution(0.0, 4.0), optimize=True)
    endo_latent: int = int_field(default=4, domain=IntDistribution(1, 16), optimize=True)
    exo_latent: int = int_field(default=4, domain=IntDistribution(1, 16), optimize=True)
    num_groups: int = int_field(default=2, domain=IntDistribution(1, 8), optimize=True)

    def problem(self) -> IVProblem:
        return IVProblem(alpha=self.alpha, beta=self.beta, gamma=self.gamma, delta=self.delta)

    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        """Model creator for the IV problem.

        Creates a single AFP node that receives z, t, y observations and runs adversarial
        factor purification.  The instrument (z) is the exogenous input, the treatment (t)
        is the endogenous input, and the outcome (y) is the observation target.
        """
        del data_source, problem
        zero = jnp.zeros(1, dtype=jnp.complex128)
        input_node = InputNode.create(field_defaults={"z": zero, "t": zero, "y": zero})
        afp = AFP.create(
            endo_features=1,
            exo_features=1,
            obs_features=1,
            endo_latent=self.endo_latent,
            exo_latent=self.exo_latent,
            num_groups=self.num_groups,
            streams=streams,
        )
        afp_node = Node.create(
            name="afp",
            kernel=afp,
            bindings={
                "z_endo": [("input", "t")],
                "z_exo": [("input", "z")],
                "z_obs": [("input", "y")],
            },
            streams=streams,
        )
        return Model.create(
            frozendict({"input": input_node, "afp": afp_node}),
            frozendict({}),
        )
