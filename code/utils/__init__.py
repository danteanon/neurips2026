# Lazy imports using __getattr__ to avoid loading heavy dependencies at import time
from utils.logging_config import setup_logging

def __getattr__(name):
    """
    Lazy loading of utility modules to avoid importing heavy dependencies
    until they are actually needed.
    """
    if name == "gpu_utils":
        import utils.gpu_utils
        return utils.gpu_utils
    elif name == "metrics":
        import utils.metrics
        return utils.metrics
    elif name == "normalisations":
        import utils.normalisations
        return utils.normalisations
    elif name == "transformations":
        import utils.transformations
        return utils.transformations
    elif name == "schedulers":
        import utils.schedulers
        return utils.schedulers
    elif name == "mlflow_model_loader":
        import utils.mlflow_model_loader
        return utils.mlflow_model_loader
    elif name == "dino_weights":
        import utils.dino_weights
        return utils.dino_weights
    elif name == "mlflow_image_logging_callback":
        import utils.mlflow_image_logging_callback
        return utils.mlflow_image_logging_callback
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Expose utility modules at the package level for direct import
__all__ = ["gpu_utils", "metrics", "normalisations", "transformations", "schedulers", 
           "mlflow_model_loader", "dino_weights", "mlflow_image_logging_callback"]

setup_logging()
