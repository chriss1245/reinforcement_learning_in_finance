"""
This module contains the custom types used in the project.
"""


from enum import Enum


class Phase(Enum):
    """
    Enum for the phase of the training.

    Attributes:
        TRAIN: Training phase
        VALID: Validation phase
        TRAIN_VALID: Training and validation (prior to testing)
        TEST: Testing phase
    """

    TRAIN = "train"
    VAL = "val"
    TRAIN_VAL = "train_val"
    TEST = "test"


class Device(Enum):
    """
    Enum for the device to use.

    Attributes:
        CPU: CPU
        GPU: GPU
    """

    CPU = "cpu"
    GPU = "cuda"
