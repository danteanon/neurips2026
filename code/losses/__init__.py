# Expose loss_fun as a submodule
from . import loss_fun
# ...add more exports as needed...
from .loss_fun import SoftBCEWithLogitsLoss, DiceLoss
from .vicreg_loss import VICRegLoss, VICRegExpander