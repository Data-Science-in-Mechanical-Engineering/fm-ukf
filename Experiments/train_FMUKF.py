"""
FMUKF Training Script

This module provides the main training pipeline for the FMUKF (Fourier-Masked Unscented Kalman Filter) model.
The script handles dataset loading, model initialization, training with PyTorch Lightning, and logging with Comet ML.

The training process includes:
1. Dataset initialization with train/val/test splits
2. Model setup with transformer architecture
3. Training with various callbacks (checkpointing, visualization, etc.)
4. Logging and experiment tracking

Usage:
    python train_FMUKF.py
    
    # With custom config
    python train_FMUKF.py data.batch_size=64 training.max_epochs=50
    
    # With different dataset
    python train_FMUKF.py data.h5_dataset_path=/path/to/dataset.h5

Dependencies:
    - PyTorch Lightning for training orchestration
    - Comet ML for experiment tracking
    - Hydra for configuration management
    - Custom fmukf package for models and data loading
"""

import os
import fmukf

from omegaconf import OmegaConf
import comet_ml # Needs to be imported before torch (even if not directly used)
from lightning.pytorch.loggers import CometLogger
import lightning as L
import torch

# Import hydra with custom resolves
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from fmukf.utils.hydra import register_fmukf_resolvers
register_fmukf_resolvers()


def setup_comet_logger(cfg) -> CometLogger:
    """
    Set up and configure Comet ML logger for experiment tracking.
    
    This function initializes the Comet ML logger with the provided configuration,
    sets up experiment metadata, logs configuration files, and registers hyperparameters
    for tracking during training.
    
    Args:
        cfg (DictConfig): Configuration object containing logger settings
        
    Returns:
        CometLogger: Configured Comet ML logger instance
        
    Raises:
        Exception: Various exceptions may occur during logger setup, but they are caught
                  and logged as warnings to prevent training interruption
    """
    # Initialize Comet ML logger with API credentials and project settings
    comet_logger = CometLogger(
        api_key=cfg.logger.comet.api_key,
        project_name=cfg.logger.comet.project_name,
        workspace=cfg.logger.comet.workspace,
        save_dir=cfg.logger.comet.save_dir,
    )
    # comet_logger = setup_comet_logger(cfg)
    experiment_name = comet_logger.experiment.get_name()

    # Add tags to the experiment for better organization
    try:
        comet_logger.experiment.add_tags(list(cfg.logger.comet.tags))
    except Exception as e:
        print(f"Couldn't add tags to the experiment because of reason: {e}")
    
    # Log the config yaml file for experiment reproducibility
    try:
        # create folder if it doesn't exist
        os.makedirs(os.path.join(cfg.logger.comet.save_dir, "temp"), exist_ok=True)
        cfg_file_path = os.path.join(cfg.logger.comet.save_dir, "temp/cfg.yaml")
        with open(cfg_file_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        comet_logger.experiment.log_code(cfg_file_path)
    except Exception as e:
        print(f"Couldn't log the configuration because of reason: {e}")

    # Log code files to the experiment for version control
    try:
        for path in cfg.logger.comet.log_code_files:
            comet_logger.experiment.log_code(file_name=path)
    except Exception as e:
        print(f"Couldn't log code files because of reason: {e}")


    # Log hyperparameters for experiment tracking
    try:
        # Main training hyperparameters
        comet_logger.experiment.log_parameter("batch_size", cfg.data.batch_size)
        comet_logger.experiment.log_parameter("l_batch", cfg.training.l_batch)
        comet_logger.experiment.log_parameter("min_context", cfg.model.masking.min_context)
        comet_logger.experiment.log_parameter("max_context", cfg.model.masking.max_context)
        comet_logger.experiment.log_parameter("learning_rate", cfg.model.optimization.learning_rate)    
        comet_logger.experiment.log_parameter("optimizer", cfg.model.optimization.optimizer)
        comet_logger.experiment.log_parameter("gradient_clip_val", cfg.training.gradient_clip_val)
        comet_logger.experiment.log_parameter("data_seed", cfg.data.seed)
        comet_logger.experiment.log_parameter("dataset", cfg.data.h5_dataset_path)
        comet_logger.experiment.log_parameter("num_envs_val", cfg.data.num_envs_val)
        comet_logger.experiment.log_parameter("num_envs_test", cfg.data.num_envs_test)
        comet_logger.experiment.log_parameter("normalize", cfg.model.normalize)
        
        # Transformer architecture parameters
        for k, v in cfg.model.transformer_kwargs.items():
            comet_logger.experiment.log_parameter(k, v)
        comet_logger.experiment.log_parameter("patch_size", cfg.model.patch_size)

        # Learning rate schedule parameters
        for k,v in cfg.model.optimization.lr_schedule.items():
            comet_logger.experiment.log_parameter("lr_schedule__"+k, v)
    
        # Noise parameters for state variables
        for k,v in cfg.model.noise_stds_x.items():
            comet_logger.experiment.log_parameter("noise_stds_x__"+k, v)
        for k,v in cfg.model.noise_stds_x.items():
            comet_logger.experiment.log_parameter("noise_stds_u__"+k, v)
        
    except Exception as e:
        print(f"Couldn't log hyperparameters because of reason: {e}")

    return comet_logger


@hydra.main(version_base=None, config_path="config", config_name="train_FMUKF")
def main(cfg: DictConfig):
    """
    Main training function for FMUKF model.
    
    This function orchestrates the complete training pipeline:
    1. Dataset initialization with proper train/val/test splits
    2. Logger setup (Comet ML for experiment tracking)
    3. Model initialization with transformer architecture
    4. Callback configuration (checkpointing, visualization, etc.)
    5. Trainer setup and model fitting
    6. Final model saving and logging
    
    Args:
        cfg (DictConfig): Configuration object containing all training parameters
                         Loaded from config/train_FMUKF.yaml and overrides
    """
    print("Starting outer main")
    print(OmegaConf.to_yaml(cfg))

    # Optionally print the config for verification
    print("Reached Config Verification")

    # -------------------------
    # Dataset initialization
    # -------------------------
    # Convert the dtype string (e.g., "float32") to the corresponding torch type.
    dtype = getattr(torch, cfg.data.dtype)
    
    # Initialize the H5 data module with train/val/test splits
    # This handles data loading, preprocessing, and batch generation
    dm = fmukf.ML.dataloader.H5LightningDataModule(
        h5_file_path_str=cfg.data.h5_dataset_path,
        num_envs_val=cfg.data.num_envs_val,
        num_envs_test=cfg.data.num_envs_test,
        seed=cfg.data.seed,
        batch_size=cfg.data.batch_size,
        seq_len=cfg.training.l_batch + 1,  # +1 for target prediction
        recompute_statistics=cfg.data.recompute_statistics,
        dtype=dtype,
    )
    print("Reached Dataset Initialization")


    # -------------------------
    # Logger initialization
    # -------------------------
    use_comet = cfg.logger.use_comet
    if use_comet:
        # Set up Comet ML logger for experiment tracking
        comet_logger = setup_comet_logger(cfg)
        experiment_name = comet_logger.experiment.get_name()
    else:
        comet_logger = None
        experiment_name = "no_comet"
    print("Reached Logger Initialization")

    # -------------------------
    # Model definition
    # -------------------------

    # Normalization parameters are computed from the dataset statistics (and and overrides for some cases)
    if cfg.model.normalize:
        # Use dataset statistics for normalization of state variables
        normalization_params_x=dict(
                u=dm.statistics["state"]["u"],
                v=dm.statistics["state"]["v"],
                r=dm.statistics["state"]["r"],
                x=dm.statistics["state"]["x"],
                y=dm.statistics["state"]["y"],
                p=dm.statistics["state"]["p"],
                phi=dm.statistics["state"]["phi"],
                delta=dm.statistics["state"]["delta"],
                n=dm.statistics["state"]["n"],
                psi_sin=(0, 1),  # (mean, std) - trigonometric functions are already normalized
                psi_cos=(0, 1),  # (mean, std) - trigonometric functions are already normalized
            )
        
        # Use same statistics for input variables (delta and n)
        normalization_params_u=dict(
            delta=dm.statistics["input"]["delta"], #<--- Yes using same as state!
            n=dm.statistics["input"]["n"],         #<--- Yes using same as state!
        )
    else:
        normalization_params_x=None
        normalization_params_u=None

    # Initialize the transformer model with all configuration parameters
    transformer_kwargs = dict(cfg.model.transformer_kwargs)
    model = fmukf.ML.models.MyTimesSeriesTransformer(
        patch_size=cfg.model.patch_size,
        transformer_kwargs=transformer_kwargs,
        datatype= cfg.model.datatype,
        h = cfg.model.h,  # Time step size
        loss_type = cfg.model.loss_type,
       
        normalization_params_u = normalization_params_u,
        normalization_params_x = normalization_params_x,
        min_std_multiplier     = cfg.model.min_std_multiplier,
        max_std_multiplier     = cfg.model.max_std_multiplier,
        noise_stds_x=cfg.model.noise_stds_x,  # Noise added to state variables
        noise_stds_u=cfg.model.noise_stds_u,  # Noise added to input variables
        masking_min_context=cfg.model.masking.min_context,  # Minimum context length for masking
        masking_max_context=cfg.model.masking.max_context,  # Maximum context length for masking
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

    # Checkpoint callback - saves best models during training
    if cfg.callbacks.checkpoint.use:
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=saved_models_dir,
            filename=cfg.callbacks.checkpoint.filename.format(experiment_name=experiment_name),
            monitor=cfg.callbacks.checkpoint.monitor,  # Monitor validation loss
            save_top_k=cfg.callbacks.checkpoint.save_top_k,  # Save top k models
            verbose=cfg.callbacks.checkpoint.verbose,
        )
        callbacks.append(checkpoint_callback)

    # Compute Identity Loss - sanity check callback
    if cfg.callbacks.identity_loss.use:
        identity_loss_callback = fmukf.ML.logging_utils.callback_IdentityLoss()
        callbacks.append(identity_loss_callback)

    # Visualization callback for raw predictions (ABBpred)
    callbacks.append(fmukf.ML.logging_utils.callback_visABBpred(
        every_n_epochs=cfg.callbacks.visABBpred.every_n_epochs,
        at_train_end=cfg.callbacks.visABBpred.at_train_end,
        backend=cfg.callbacks.visABBpred.backend,
    ))

    # Visualization callback for model unrolling and MLHP
    integrator_methods = list(cfg.callbacks.visUnroll.integrator_methods)
    callbacks.append(fmukf.ML.logging_utils.callback_visUnroll(
                    every_n_epochs=cfg.callbacks.visUnroll.every_n_epochs,
                at_train_end=cfg.callbacks.visUnroll.every_n_epochs,
                do_MLHP=cfg.callbacks.visUnroll.every_n_epochs,  # Maximum Likelihood Hyperparameter
                integrator_methods = integrator_methods if len(integrator_methods) > 0 else None,
                backend = cfg.callbacks.visUnroll.backend,
                    ))

    # Gradient accumulation callback - for effective larger batch sizes
    callbacks.append(L.pytorch.callbacks.GradientAccumulationScheduler(
        scheduling=dict(cfg.callbacks.grad_accumulation.schedule)
    ))
    print("Reached Callbacks Configuration")

    # -------------------------
    # Trainer and fitting
    # -------------------------
    # Initialize PyTorch Lightning trainer with all configurations
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_progress_bar=cfg.training.enable_progress_bar,
        gradient_clip_val=cfg.training.gradient_clip_val,  # Gradient clipping for stability
        benchmark=cfg.training.benchmark,  # Benchmark for performance optimization
        callbacks=callbacks,
        logger=comet_logger,
    )
    print("Reached Trainer Initialization")

    # Start the training process
    trainer.fit(model, dm)
    print("Reached Trainer Fitting")

    # -------------------------
    # Final saving of the model
    # -------------------------
    # Save the final trained model
    final_model_path = os.path.join(cfg.output_dir, f"fmukf_{experiment_name}.ckpt")
    trainer.save_checkpoint(final_model_path)
    if use_comet:
        comet_logger.experiment.log_model("final_model", final_model_path)

    # Log all temporary top-k models with comet for experiment tracking
    if use_comet:
        try:
            for root, dirs, files in os.walk(saved_models_dir):
                for file in files:
                    if file.endswith(".ckpt"):
                        model_path = os.path.join(root, file)
                        comet_logger.experiment.log_model(file, model_path)
        except Exception as e:
            print(f"Couldn't log saved models because of reason: {e}")


if __name__ == "__main__":
    main()
