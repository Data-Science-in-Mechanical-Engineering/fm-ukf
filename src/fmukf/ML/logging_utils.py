import lightning as L
from fmukf.ML.models import MyTimesSeriesTransformer
import torch
import os
import holoviews as hv
import numpy as np
from lightning.pytorch.loggers import CometLogger

def log_hv(logger: CometLogger, hvLayout: 'hv.core.layout.Layout', filename: str = "vis.png"):
    """
    Logs a HoloViews layout to a logging system (currently only supports Comet logger).
    
    Saves the visualization as either a PNG or HTML file and logs it to the experiment
    tracking system. For PNG files, uses matplotlib backend; for HTML files, uses bokeh backend.
    
    Args:
        logger: lightning.LightningLoggerBase
            The logger instance to use for logging. Currently only supports Comet logger.
        hvLayout: hv.core.layout.Layout
            The HoloViews layout object to be logged and saved.
        filename: str, default="vis.png"
            The filename where the visualization will be saved. Determines the file format
            and backend used (PNG for matplotlib, HTML for bokeh).
    """
    print("log_hv: start")
    try:
        assert "comet" in str(type(logger)), f"Only comet logger is supported. Will not log image '{filename}'"

        # Create folder if not exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        filetype = filename.split(".")[-1]
        if filetype == "png":
            import matplotlib
            matplotlib.use('Agg')  # Must be set before any matplotlib-related imports
            hv.extension("matplotlib")
            
            print("log_hv: about to save matplotlib ", filename)
            hv.save(hvLayout, filename, backend="matplotlib")
            print("log_hv: saved, now about to log ", filename)

            assert os.path.exists(filename), f"File '{filename}' does not exist"
            print("does file exist: ", os.path.exists(filename))
            print("full path: ", os.path.abspath(filename))

            logger.experiment.log_image(filename)
            print("log_hv: saved, now about to log ", filename)
        elif filetype == "html":
            hv.extension("bokeh")
            print("log_hv: about to save bokeh ", filename)
            hv.save(hvLayout, filename, backend="bokeh")
            with open(filename, "r") as file:
                print("log_hv: about to save bokeh ", filename)
                html_str = file.read()
                logger.experiment.log_html(html_str)
    except Exception as e:
        print(f"Couln't' log image {filename} because of reason: {e}")


import holoviews as hv
import numpy as np
def visABBpred_(model: 'MyTimesSeriesTransformer', batch: tuple, backend: str = "matplotlib") -> hv.Layout:
    """
    Creates a visualization comparing raw input sequences (A), target sequences (B), and model predictions (Bpred)
    
    Generates subplots for each variable in the sequence, showing the input sequence, target sequence,
    and model predictions. Filters out NaN values from predictions that occur due to input patching.
    
    Args:
        model: MyTimesSeriesTransformer
            The trained transformer model to use for making predictions.
        batch: tuple
            The input batch containing (x, u) where x is the state sequence and u is the control sequence.
            x: np.ndarray of shape (batch_size, time_steps, features) - State sequences
            u: np.ndarray of shape (batch_size, time_steps, control_features) - Control sequences
        backend: str, default="matplotlib"
            The HoloViews backend to use for rendering ("matplotlib" or "bokeh").
    
    Returns:
        hv.Layout: A HoloViews layout containing subplots for each variable showing input, target, and prediction sequences.
    
    Example:
        >>> vis = visABBpred_(model, batch, backend="matplotlib")
        >>> hv.save(vis, "prediction_comparison.png")
    """
    print("visABBpred_: start")

    import holoviews as hv
    hv.extension(backend)
    # x, u = batch

    # Compute Target and make prediction
    # A, B = model.prepare_input_target(x, u)
    A, B = model.prepare_input_target(batch)
    Bpred = model.forward_with_nans(A)

    # Put back to cpu and numpy
    A_ = A[0].detach().cpu().numpy()
    B_ = B[0].detach().cpu().numpy()
    Bpred_ = Bpred[0].detach().cpu().numpy()

    # Do visualization
    subplots = []
    for dim, var_name in enumerate(["u", "v", "r", "x", "y", "p", "phi", "delta", "n", "psi_sin", "psi_cos", "ctrl_delta", "ctrl_n"]):
        
        # Target B
        if dim < B_.shape[1]:
            subplot = hv.Curve(B_[:,dim], label="B")

        # Prediction Bpred
        if dim < Bpred_.shape[1]:
            t_idx_not_nan = ~np.isnan(Bpred_[:,dim])   # Filter out values with nans (these are the time-steps not predicted due to input-patching)
            t_ = np.arange(len(Bpred_))[t_idx_not_nan]
            Bpred__ = Bpred_[t_idx_not_nan,dim]
            subplot *= hv.Curve((t_, Bpred__), label="Bpred")

        # Input Sequence A
        subplot *= hv.Curve(A_[:,dim], label="A")
        subplots.append(subplot.opts(title=var_name))

    vis = hv.Layout(subplots).opts(shared_axes=False)

    print("visABBpred_: end")
    return vis


class callback_visABBpred(L.Callback):
    """
    Lightning callback for visualizing input-target-prediction sequences during training.
    
    This callback creates visualizations comparing the model's input sequences (A), target sequences (B),
    and predicted sequences (Bpred) at specified intervals during training. The visualizations are logged
    to the experiment tracking system.
    
    The callback can be configured to run at specific epoch intervals and/or at the end of training.
    """
    
    def __init__(self, every_n_epochs: int = 5, at_train_end: bool = False, backend="matplotlib"):
        """
        Initialize the visualization callback.
        
        Args:
            every_n_epochs: int, default=5
                The frequency (in epochs) at which to perform visualization of the input and output sequences.
                Set to 0 to disable periodic visualization.
            at_train_end: bool, default=False
                Whether to perform visualization at the end of training.
            backend: str, default="matplotlib"
                The HoloViews backend to use for rendering ("matplotlib" or "bokeh").
        """
        self.every_n_epochs = every_n_epochs
        self.at_train_end = at_train_end
        self.backend = backend

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Check if the current epoch is one where visualization should run
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            # Access the logger via trainer.logger
            self.visualize_and_log(trainer, pl_module)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.at_train_end:
            self.visualize_and_log(trainer, pl_module)

    def visualize_and_log(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger = trainer.logger
        
        # Access the model (pl_module) as needed for visualization
        model:'MyTimesSeriesTransformer' = pl_module
        
        # Get one batch from the first validation dataloader.
        # Note: trainer.val_dataloaders can be a list; here we use the first one.
        val_loader = trainer.val_dataloaders
        batch = next(iter(val_loader))
        
        # Now perform visualization
        vis = visABBpred_(model, batch, backend=self.backend)
        log_hv(logger, vis, filename=f"temp/vis/visABBpred.{'html' if self.backend=='bokeh' else 'png'}")

class callback_IdentityLoss(L.Callback):
    """
    Lightning callback for computing the "identity loss" at the beginning of training.

    The identity loss is defined as mean loss when using the current (encoded/transformed) state x_k as the
    prediction for x_k+1. This serves as a very simple baseline.
    
    This callback calculates the identity loss across all validation batches at the beginning of training
    and logs the results to provide a baseline performance metric before actual model training begins.
    """
    
    def __init__(self):
        """
        Initialize the identity loss callback.
        """
        pass

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Called at the start of training.
        
        Computes identity loss across all validation batches and logs the results.
        The identity loss serves as a baseline to compare against the model's actual performance.
        
        Args:
            trainer: L.Trainer
                The Lightning trainer instance.
            pl_module: L.LightningModule
                The Lightning module (model) being trained.
        """
        # Access the logger via trainer.logger
        print("Doing identity loss")
        # logger = trainer.logger
        
        # Access the model (pl_module) as needed for visualization
        model:'MyTimesSeriesTransformer' = pl_module
        
        val_loader = trainer.val_dataloaders
        # Loop through all batches
        for batch in val_loader:
            # Now perform visualization
            A,B = model.prepare_input_target(batch)
            loss = model.loss_fn(B[:,1:,:], B[:,:-1,:])
            print(loss)
            self.log("IdentityLoss", loss, on_epoch=True, on_step=False)

        print("Done With Identity Loss")

class callback_visUnroll(L.Callback):
    """
    Lightning callback for visualizing model's predictions during training.

    Here the model is fed with an initial sequence {x_k; k=1,...,L_context} of true states
    and asked to predict the rest of the trajectory using various methods:
    - iterative Unrolling the model using its own predictions as the context
    - Masked Long Horizon Prediction (MLHP): predicting all future states in one go, by
       masking the future states L>L_context with zero-values. (this is what is used during training)
      
    """
    
    def __init__(self,
                 every_n_epochs: int = 5,
                 at_train_end: bool = True,
                 max_context: int = None,
                 do_MLHP: bool = True,
                 integrator_methods: list[str] = None,
                 backend: str = "matplotlib"):
        """
        Initialize the unroll visualization callback.
        
        Args:
            every_n_epochs: int, default=5
                The frequency (in epochs) at which to perform visualization of trajectory predictions.
                Set to 0 to disable periodic visualization.
            at_train_end: bool, default=True
                Whether to perform visualization at the end of training.
            max_context: int, default=None
                Maximum number of time steps to use as context for predictions, and also L_context in the paper.
                If None, uses model's masking_min_context value.
            do_MLHP: bool, default=True
                Whether to include Masked Long Horizon Prediction in the visualization.
            integrator_methods: list[str], default=None
                List of integrator methods to use for unrolling (e.g., ["euler", "rk4"]).
            backend: str, default="matplotlib"
                The HoloViews backend to use for rendering ("matplotlib" or "bokeh").
        """

        self.every_n_epochs = every_n_epochs
        self.at_train_end = at_train_end
        self.max_context = max_context
        self.masked_prediction = do_MLHP
        self.integrator_methods = integrator_methods
        self.backend = backend
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Check if the current epoch is one where visualization should run
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.visualize_and_log(trainer, pl_module)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.at_train_end:
            self.visualize_and_log(trainer, pl_module)

    def visualize_and_log(self, trainer: L.Trainer, pl_module: L.LightningModule):
        print("Visualizing Unroll")
        # Access the logger via trainer.logger
        logger = trainer.logger
        
        # Access the model (pl_module) as needed for visualization
        model:'MyTimesSeriesTransformer' = pl_module
        
        # Get one batch from the first validation dataloader.
        # Note: trainer.val_dataloaders can be a list; here we use the first one.
        val_loader = trainer.val_dataloaders
        batch = next(iter(val_loader))
        
        # Now perform visualization
        try:
            vis = self.visUnroll(model,
                                batch,
                                max_context=self.max_context,
                                do_MHLP=self.masked_prediction,
                                integrator_methods=self.integrator_methods,
                                backend=self.backend)
            log_hv(logger, vis, filename=f"temp/vis/visUnroll.{'html' if self.backend=='bokeh' else 'png'}")
        except Exception as e:
            print(f"Couldn't visualize unroll because of reason: {e}")
            pass

    @staticmethod
    def visUnroll(model: 'MyTimesSeriesTransformer',
                  batch: tuple,
                  max_context: int = None,
                  do_MHLP: bool = True,
                  integrator_methods: list[str] = None,
                  backend: str = "matplotlib") -> hv.Layout:
        """
        Create visualization of model prediction unrolling.
        
        Visualizes the model's performance by comparing different prediction methods:
        1. Standard unrolling (using the model's own predictions as context)
        2. Integrator-based unrolling (if integrator_methods are provided)
        3. Masked Long Horizon Prediction (MLHP) if enabled
        
        Args:
            model: MyTimesSeriesTransformer
                The trained transformer model to use for making predictions.
            batch: tuple
                The input batch containing (x, u) where:
                x: np.ndarray or torch.Tensor of shape (batch_size, time_steps, features) - State sequences
                u: np.ndarray or torch.Tensor of shape (batch_size, time_steps, control_features) - Control sequences
            max_context: int, default=None
                Number of initial time steps to use as context for predictions.
                If None, uses model's masking_min_context value.
            do_MHLP: bool, default=True
                Whether to include Masked Long Horizon Prediction in the visualization.
            integrator_methods: list[str], default=None
                List of integrator methods to use for unrolling (e.g., ["euler", "rk4"]).
            backend: str, default="matplotlib"
                The HoloViews backend to use for rendering ("matplotlib" or "bokeh").
                
        Returns:
            hv.Layout: A HoloViews layout containing subplots for each state variable showing the
                      different prediction methods compared to the ground truth.
        """
        # Import holoviews here to avoid circular import
        import holoviews as hv
        hv.extension(backend)

        if max_context is None:
            max_context = model.masking_min_context
        print("Visualizing Unrolling with max_context", max_context)
        

        # Convert to tensors, add in batch dimension if needed
        x,u = batch
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=model.dtype_).to(model.device)
            u = torch.tensor(u, dtype=model.dtype_).to(model.device)
        if len(x.shape) == 3:
            x = x[0]
            u = u[0]
        elif len(x.shape) == 1:
            raise ValueError(f"Input x should have shape (batch, time, features) but got shape (time, features)")
        
        # Wrap heading angle to be between 0 and 360 degrees
        x[:,5] = x[:,5] % 360

        # Unrolling (no integration)
        uin = u[:-1, :].unsqueeze(0) # <-- Remove last element to match shape of x after unrolling
        xin = x[:max_context, :]
        xpred_unroll = model.unroll(xin, uin, integrator_method=None, max_context=max_context)
        xpred_unroll = xpred_unroll.detach().cpu().numpy()

        # Unrolling (with integration)
        xpreds_integrated = {}
        if integrator_methods is not None:
            for integrator_method in integrator_methods:
                # xpreds_integrated[integrator_method] = model.unroll(xin, uin, integrator_method=integrator_method)
                # print(integrator_method)
                try:
                    xpreds_integrated[integrator_method] = model.unroll(xin, uin, integrator_method=integrator_method)
                except Exception as e:
                    print(f"Couldn't integrate with method '{integrator_method}' because of reason: {e}")

        # Masked Long Horizon Prediction
        if do_MHLP:
            xpred_masked = model.predict_with_nans(xin, uin, max_context=max_context)
            xpred_masked = xpred_masked.detach().cpu().numpy()
        
        # Visualize
        subplots = []        
        for dim, var_name in enumerate(["u", "v", "r", "x", "y", "psi", "p", "phi", "delta", "n"]):
            
            # Grey Box to indicate Context
            opts = dict(facecolor="grey",edgecolor="grey", alpha=0.2) if backend=="matplotlib" else dict(color="grey", alpha=0.2)
            subplot = hv.VSpan(0,max_context-1).opts(**opts)
            subplot *= hv.VLine(max_context).opts(color="grey") #, line_dash="dashed")
            
            # Target 
            subplot *= hv.Curve(x[:,dim], label="Target")

            # Unrolled Prediction (No Integration)
            subplot *= hv.Curve(xpred_unroll[:,dim], label="Prediction (Unrolling)")

            # Unrolled Prediction (With Integration)
            for integrator_method, xpred_integrated in xpreds_integrated.items():
                subplot *= hv.Curve(xpred_integrated[:,dim], label=f"Prediction ({integrator_method})")
            
            # Masked Long Horizon Prediction
            if do_MHLP:
                # Only plot time-steps where is not none
                t = np.arange(len(xpred_masked))
                tidx_notnan = ~np.isnan(xpred_masked[:,dim])
                subplot *= hv.Curve((t[tidx_notnan], xpred_masked[tidx_notnan,dim]), label="Prediction (MLHP)")
            # Format Legend
            if dim == 0:
                subplots.append(subplot.opts(title=var_name, legend_position="top_left"))
            else:
                subplots.append(subplot.opts(title=var_name, show_legend=False))
        vis = hv.Layout(subplots).opts(shared_axes=False)

        # Remove annoying sublabels in matplotlib rendering option
        if backend == "matplotlib":
            vis = vis.opts(sublabel_format="")
        return vis