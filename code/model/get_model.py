import model as model_classes

def get_model(config):
    """
    Initialize the segmentation model based on configuration

    Args:
        config (dict): Configuration dictionary

    Returns:
        torch.nn.Module: The initialized model
    """
    # Updated to support nested model config
    model_type = getattr(model_classes, config["model"]["type"])
    model_name = config["model"]["name"]
    model = getattr(model_type, model_name)(**config[model_name])
    return model
