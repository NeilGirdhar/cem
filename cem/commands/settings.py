from pathlib import Path

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from cem.structure.solution.wandb_tools import WAndBInitSettings

jax_cache_dir = "./.jax_cache"
optuna_cache = Path("./.optuna_cache/journal.log").resolve()
optuna_sampler = None
wandb_cache_path = Path("./.wandb_cache").resolve()
wandb_settings = WAndBInitSettings(dir=wandb_cache_path, entity="neilgirdhar")


def get_optuna_storage() -> JournalStorage:
    optuna_cache.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(optuna_cache.as_posix()))
