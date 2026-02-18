import torch
import segmentation_models_pytorch as smp

def build_model(num_classes: int, encoder_name: str = "resnet34", encoder_weights: str = "imagenet"):
    """
    Returns a segmentation model that outputs logits (B, C, H, W).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,  # IMPORTANT: logits, not softmax
    )
    return model
