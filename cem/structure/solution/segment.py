from __future__ import annotations

import jax.random as jr
from tjax import KeyArray


def segment_keys(key: KeyArray, episodes: int) -> tuple[KeyArray, KeyArray]:
    """Produce random keys for every episode for generating examples and inference.

    Args:
        key: KeyArray  # For example and inference.
        episodes: int  # The number of RL episodes to run.
    """
    example_key_base, inference_key_base = jr.split(key)
    example_keys = jr.split(example_key_base, episodes)
    inference_keys = jr.split(inference_key_base, episodes)
    return example_keys, inference_keys
