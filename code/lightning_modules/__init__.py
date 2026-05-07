# Previously this file eagerly imported `data_module`, `model_module`, and
# `sae_module` as a side effect of importing any single sibling
# (e.g. `from lightning_modules.model_module import SegmentationModule`).
# That pulled the full data-pipeline import chain into memory on every
# training run, including the parquet + label-studio subsystems that
# height estimation never touches. The chain also caused SageMaker
# training to crash at container start whenever any of those indirectly
# required packages were missing or excluded from the image.
#
# Every caller in the repo uses direct submodule imports
# (`from lightning_modules.X import Y`), so this __init__.py can stay empty.
