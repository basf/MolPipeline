"""Module for accessing the parameters of a lightning trainer."""

from pathlib import Path
from typing import Any

try:
    import lightning as pl
    import torch
    from lightning.pytorch.accelerators import CPUAccelerator, CUDAAccelerator
    from lightning.pytorch.callbacks import ModelSummary, ProgressBar, Timer
    from lightning.pytorch.trainer.connectors.accelerator_connector import Accelerator
except ImportError:
    pass


def get_enable_progress_bar(trainer: pl.Trainer) -> bool:
    """Get if the progress bar is enabled in the lightning trainer.

    Parameters
    ----------
    trainer : pl.Trainer
        The lightning trainer.

    Returns
    -------
    bool
        If the progress bar is enabled in the lightning trainer.
    """
    for callback in trainer.callbacks:  # type: ignore[attr-defined]
        if isinstance(callback, ProgressBar):
            return True
    return False


def get_device(trainer: pl.Trainer) -> str | Accelerator:
    """Get the device used by the lightning trainer.

    Parameters
    ----------
    trainer : pl.Trainer
        The lightning trainer.

    Returns
    -------
    str
        The device used by the lightning trainer.
    """
    if isinstance(trainer.accelerator, CPUAccelerator):
        devices = "cpu"
    elif isinstance(trainer.accelerator, CUDAAccelerator):
        devices = "gpu"
    else:
        devices = trainer.accelerator  # type: ignore[attr-defined]
    return devices


TRAINER_DEFAULT_PARAMS = {
    "strategy": "auto",
    "limit_predict_batches": 1.0,
    "fast_dev_run": False,
    "min_steps": None,
    "accumulate_grad_batches": 1,
    "use_distributed_sampler": True,
    "devices": "auto",
    "check_val_every_n_epoch": 1,
    "enable_progress_bar": True,
    "max_epochs": None,
    "max_time": None,
    "val_check_interval": 1.0,
    "log_every_n_steps": 50,
    "min_epochs": None,
    "gradient_clip_algorithm": None,
    "max_steps": -1,
    "limit_val_batches": 1.0,
    "gradient_clip_val": None,
    "inference_mode": True,
    "enable_model_summary": None,
    "limit_test_batches": 1.0,
    "reload_dataloaders_every_n_epochs": 0,
    "deterministic": False,
    "logger": None,
    "overfit_batches": 0.0,
    "precision": "32-true",
    "benchmark": False,
    "num_sanity_val_steps": 2,
    "enable_checkpointing": None,
    "limit_train_batches": 1.0,
    "barebones": False,
    "detect_anomaly": False,
    "num_nodes": 1,
}


def get_trainer_path(trainer: pl.Trainer) -> str | None:
    """Get the path of the lightning trainer.

    Parameters
    ----------
    trainer : pl.Trainer
        The lightning trainer.

    Returns
    -------
    str | None
        The path of the lightning trainer.
        None if the path is the current path.
    """
    curr_path = str(Path(".").resolve())
    trainer_path = trainer.default_root_dir
    if trainer_path == curr_path:
        trainer_path = None
    return trainer_path


def get_params_trainer(trainer: pl.Trainer) -> dict[str, Any]:
    """Get the parameters of the lightning trainer.

    Parameters
    ----------
    trainer : pl.Trainer
        The lightning trainer.

    Returns
    -------
    dict[str, Any]
        The parameters of the lightning trainer.
    """
    if trainer.callbacks and isinstance(trainer.callbacks[-1], Timer):  # type: ignore[attr-defined]
        max_time = trainer.callbacks[-1].duration  # type: ignore[attr-defined]
    else:
        max_time = None

    for callback in trainer.callbacks:  # type: ignore[attr-defined]
        if isinstance(callback, ModelSummary):
            enable_progress_model_summary = True
            break
    else:
        enable_progress_model_summary = False

    trainer_dict = {
        "accelerator": get_device(trainer),
        "strategy": "auto",  # trainer.strategy, # collides with accelerator
        "devices": "auto",  # trainer._accelerator_connector._devices_flag does not really work
        "num_nodes": trainer.num_nodes,
        "precision": trainer.precision,
        "logger": trainer.logger,
        # "callbacks": trainer.callbacks,  # type: ignore[attr-defined]
        "fast_dev_run": trainer.fast_dev_run,  # type: ignore[attr-defined]
        "max_epochs": trainer.max_epochs,
        "min_epochs": trainer.min_epochs,
        "max_steps": trainer.max_steps,
        "min_steps": trainer.min_steps,
        "max_time": max_time,
        "limit_train_batches": trainer.limit_train_batches,
        "limit_val_batches": trainer.limit_val_batches,
        "limit_test_batches": trainer.limit_test_batches,
        "limit_predict_batches": trainer.limit_predict_batches,
        "overfit_batches": trainer.overfit_batches,  # type: ignore[attr-defined]
        "val_check_interval": trainer.val_check_interval,
        "check_val_every_n_epoch": trainer.check_val_every_n_epoch,
        "num_sanity_val_steps": trainer.num_sanity_val_steps,
        "log_every_n_steps": trainer.log_every_n_steps,  # type: ignore[attr-defined]
        "enable_checkpointing": bool(trainer.checkpoint_callbacks),
        "enable_progress_bar": get_enable_progress_bar(trainer),
        "enable_model_summary": enable_progress_model_summary,
        "accumulate_grad_batches": trainer.accumulate_grad_batches,
        "gradient_clip_val": trainer.gradient_clip_val,
        "gradient_clip_algorithm": trainer.gradient_clip_algorithm,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "inference_mode": trainer.predict_loop.inference_mode,
        "use_distributed_sampler": trainer._accelerator_connector.use_distributed_sampler,  # pylint: disable=protected-access
        # "profiler": trainer.profiler,  # type: ignore[attr-defined]
        "detect_anomaly": trainer._detect_anomaly,  # pylint: disable=protected-access
        "barebones": trainer.barebones,
        # "plugins": trainer.plugins,  # can not be exctracted
        # "sync_batchnorm": trainer._accelerator_connector.sync_batchnorm,  # plugin related
        "reload_dataloaders_every_n_epochs": trainer.reload_dataloaders_every_n_epochs,  # type: ignore[attr-defined]
        "default_root_dir": get_trainer_path(trainer),
    }
    return trainer_dict


def get_non_default_params_trainer(trainer: pl.Trainer) -> dict[str, Any]:
    """Get the parameters of the lightning trainer which are not default.

    Parameters
    ----------
    trainer : pl.Trainer
        The lightning trainer.

    Returns
    -------
    dict[str, Any]
        The parameters of the lightning trainer.
    """
    trainer_dict = get_params_trainer(trainer)
    non_default_values = {}
    for key, value in trainer_dict.items():
        if key not in TRAINER_DEFAULT_PARAMS:
            non_default_values[key] = value
        elif value != TRAINER_DEFAULT_PARAMS[key]:
            non_default_values[key] = value
    return non_default_values
