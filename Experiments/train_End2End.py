#!/usr/bin/env python3
"""
End2End Transformer Training Script

This script trains the End2End transformer model for state estimation using partial observations.
The model learns to reconstruct full state vectors from sensor-based partial observations.

Usage:
    python train_End2End.py

The script uses Hydra for configuration management and PyTorch Lightning for training.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# Import hydra with custom resolves
import fmukf
from fmukf.utils.hydra import register_fmukf_resolvers
register_fmukf_resolvers()

# Import visualization backend
import holoviews as hv
hv.extension('matplotlib')


def setup_comet_logger(cfg: DictConfig):
    """
    Factory function to create a CometLogger for experiment tracking.
    
    Parameters
    ----------
    cfg: DictConfig
        Configuration object containing logger settings
        
    Returns
    -------
    CometLogger
        Configured Comet ML logger for experiment tracking
    """
    from lightning.pytorch.loggers import CometLogger
    
    comet_logger = CometLogger(
        api_key=cfg.logger.comet.api_key,
        project_name=cfg.logger.comet.project_name,
        workspace=cfg.logger.comet.workspace,
        save_dir=cfg.logger.comet.save_dir,
    )
    
    experiment_name = comet_logger.experiment.get_name()

    # Add tags to the experiment
    try:
        comet_logger.experiment.add_tags(list(cfg.logger.comet.tags))
    except Exception as e:
        print(f"Couldn't add tags to the experiment because of reason: {e}")
    
    # Log the config yaml file
    try:
        # Create folder if it doesn't exist
        os.makedirs(os.path.join(cfg.logger.comet.save_dir, "temp"), exist_ok=True)
        cfg_file_path = os.path.join(cfg.logger.comet.save_dir, "temp/cfg.yaml")
        with open(cfg_file_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        comet_logger.experiment.log_code(cfg_file_path)
    except Exception as e:
        print(f"Couldn't log the configuration because of reason: {e}")

    # Log code files to the experiment
    try:
        for path in cfg.logger.comet.log_code_files:
            comet_logger.experiment.log_code(file_name=path)
    except Exception as e:
        print(f"Couldn't log code files because of reason: {e}")

    # Log hyperparameters
    try:
        # Main hyperparameters
        comet_logger.experiment.log_parameter("batch_size", cfg.data.batch_size)
        comet_logger.experiment.log_parameter("l_batch", cfg.training.l_batch)
        comet_logger.experiment.log_parameter("learning_rate", cfg.model.optimization.learning_rate)    
        comet_logger.experiment.log_parameter("optimizer", cfg.model.optimization.optimizer)
        comet_logger.experiment.log_parameter("gradient_clip_val", cfg.training.gradient_clip_val)
        comet_logger.experiment.log_parameter("data_seed", cfg.data.seed)
        comet_logger.experiment.log_parameter("dataset", cfg.data.h5_dataset_path)
        comet_logger.experiment.log_parameter("num_envs_val", cfg.data.num_envs_val)
        comet_logger.experiment.log_parameter("num_envs_test", cfg.data.num_envs_test)
        comet_logger.experiment.log_parameter("normalize", cfg.model.normalize)
        
        # Transformer parameters
        for k, v in cfg.model.transformer_kwargs.items():
            comet_logger.experiment.log_parameter(k, v)
        comet_logger.experiment.log_parameter("patch_size", cfg.model.patch_size)

        # Learning rate schedule parameters
        for k, v in cfg.model.optimization.lr_schedule.items():
            comet_logger.experiment.log_parameter("lr_schedule__"+k, v)
    
        # Noise parameters
        for k, v in cfg.model.noise_stds_x.items():
            comet_logger.experiment.log_parameter("noise_stds_x__"+k, v)
        for k, v in cfg.model.noise_stds_u.items():
            comet_logger.experiment.log_parameter("noise_stds_u__"+k, v)
        
        # Sensor configuration parameters
        comet_logger.experiment.log_parameter("num_sensor_configs", len(cfg.sensor_configs))
        for i, config in enumerate(cfg.sensor_configs):
            comet_logger.experiment.log_parameter(f"sensor_config_{i}_key", config.key)
            comet_logger.experiment.log_parameter(f"sensor_config_{i}_features", len(config.y_vec_order))
        
    except Exception as e:
        print(f"Couldn't log hyperparameters because of reason: {e}")

    return comet_logger


@hydra.main(version_base=None, config_path="config", config_name="train_End2End")
def main(cfg: DictConfig):
    """
    Main training function for the End2End transformer model.
    
    Parameters
    ----------
    cfg: DictConfig
        Configuration object containing all training parameters
    """
    print("Starting End2End training")
    print(OmegaConf.to_yaml(cfg))

    # -------------------------
    # Import required libraries
    # -------------------------
    import comet_ml
    from lightning.pytorch.loggers import CometLogger
    import lightning as L
    import torch
    from importlib import reload
    
    # Import FMUKF modules
    from fmukf.estimators.End2End import End2EndTransformer


    print("Reached Config Verification")

    # -------------------------
    # Dataset initialization
    # -------------------------
    # Convert the dtype string to the corresponding torch type
    dtype = getattr(torch, cfg.data.dtype)
    dm = fmukf.ML.dataloader.H5LightningDataModule(
        h5_file_path_str=cfg.data.h5_dataset_path,
        num_envs_val = cfg.data.num_envs_val,
        num_envs_test = cfg.data.num_envs_test,
        seed=cfg.data.seed,
        batch_size=cfg.data.batch_size,
        seq_len=cfg.training.l_batch,
        recompute_statistics=cfg.data.recompute_statistics,
        dtype=dtype,
    )
    print("Reached Dataset Initialization")

    # -------------------------
    # Logger initialization
    # -------------------------
    use_comet = cfg.logger.use_comet
    if use_comet:
        comet_logger = setup_comet_logger(cfg)
        experiment_name = comet_logger.experiment.get_name()
    else:
        comet_logger = None
        experiment_name = "no_comet"
    print("Reached Logger Initialization")

    # -------------------------
    # Model definition
    # -------------------------
    print("Setting up End2End model...")
    
    # Normalization parameters are computed from the dataset statistics
    if cfg.model.normalize:
        normalization_params_x = dict(
            u=dm.statistics["state"]["u"],
            v=dm.statistics["state"]["v"],
            r=dm.statistics["state"]["r"],
            x=dm.statistics["state"]["x"],
            y=dm.statistics["state"]["y"],
            p=dm.statistics["state"]["p"],
            phi=dm.statistics["state"]["phi"],
            delta=dm.statistics["state"]["delta"],
            n=dm.statistics["state"]["n"],
            psi_sin=(0, 1),
            psi_cos=(0, 1),
        )
        
        normalization_params_u = dict(
            delta=dm.statistics["input"]["delta"],
            n=dm.statistics["input"]["n"],
        )
    else:
        normalization_params_x = None
        normalization_params_u = None

    print("DEBUG normalize")
    print("DEBUG normalization_params_x: ", normalization_params_x)
    print("DEBUG normalization_params_u: ", normalization_params_u)

    # Prepare sensor configs for End2End training
    sensor_configs = [sensor_config["y_vec_order"] for sensor_config in cfg.sensor_configs]
    
    # Prepare transformer kwargs
    transformer_kwargs = dict(cfg.model.transformer_kwargs)
    
    # Create End2End model
    model = End2EndTransformer(
        sensor_configs=sensor_configs,
        patch_size=cfg.model.patch_size,
        transformer_kwargs=transformer_kwargs,
        datatype=cfg.model.datatype,
        h=cfg.model.h,
        loss_type=cfg.model.loss_type,
        normalization_params_u=normalization_params_u,
        normalization_params_x=normalization_params_x,
        noise_stds_x=cfg.model.noise_stds_x,
        noise_stds_u=cfg.model.noise_stds_u,
        min_std_multiplier=cfg.model.min_std_multiplier,
        max_std_multiplier=cfg.model.max_std_multiplier,
        optimizer=cfg.model.optimization.optimizer,
        learning_rate=cfg.model.optimization.learning_rate,
        lr_schedule_params=cfg.model.optimization.lr_schedule,
    )
    print("Reached Model Definition")

    # -------------------------
    # Callbacks configuration
    # -------------------------
    callbacks = []
    saved_models_dir = cfg.callbacks.checkpoint.dirpath

    # Checkpoint callback
    if cfg.callbacks.checkpoint.use:
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=saved_models_dir,
            filename=cfg.callbacks.checkpoint.filename.format(experiment_name=experiment_name),
            monitor=cfg.callbacks.checkpoint.monitor,
            save_top_k=cfg.callbacks.checkpoint.save_top_k,
            verbose=cfg.callbacks.checkpoint.verbose,
        )
        callbacks.append(checkpoint_callback)

    # Compute Identity Loss
    if cfg.callbacks.identity_loss.use:
        identity_loss_callback = fmukf.ML.logging_utils.callback_IdentityLoss()
        callbacks.append(identity_loss_callback)

    # Visualization (ABBpred raw)
    callbacks.append(fmukf.ML.logging_utils.callback_visABBpred(
        every_n_epochs=cfg.callbacks.visABBpred.every_n_epochs,
        at_train_end=cfg.callbacks.visABBpred.at_train_end,
        backend=cfg.callbacks.visABBpred.backend,
    ))

    # Gradient accumulation
    callbacks.append(L.pytorch.callbacks.GradientAccumulationScheduler(
        scheduling=dict(cfg.callbacks.grad_accumulation.schedule)
    ))
    print("Reached Callbacks Configuration")

    # -------------------------
    # Trainer and fitting
    # -------------------------
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_progress_bar=cfg.training.enable_progress_bar,
        gradient_clip_val=cfg.training.gradient_clip_val,
        benchmark=cfg.training.benchmark,
        callbacks=callbacks,
        logger=comet_logger,
    )
    print("Reached Trainer Initialization")

    trainer.fit(model, dm)
    print("Reached Trainer Fitting")

    # -------------------------
    # Final saving of the model
    # -------------------------
    final_model_path = os.path.join(saved_models_dir, f"{experiment_name}__final.ckpt")
    trainer.save_checkpoint(final_model_path)
    if use_comet:
        comet_logger.experiment.log_model("final_model", final_model_path)

    # Save all models in saved_models_dir
    try:
        for root, dirs, files in os.walk(saved_models_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    model_path = os.path.join(root, file)
                    comet_logger.experiment.log_model(file, model_path)
    except Exception as e:
        print(f"Couldn't log saved models because of reason: {e}")
    
    print("Reached Final Model Saving")


if __name__ == "__main__":
    main()
