import equinox as eqx

from cem.structure.model import Model, ModelConfiguration


class RLTrajectory(eqx.Module):
    configuration: ModelConfiguration
    model: Model
