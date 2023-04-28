"""
In-Memory CheckpointIO Plugin
===================

Automatically save in-memory model checkpoints during training.
"""

from copy import deepcopy
from lightning.pytorch.plugins import CheckpointIO


class InMemoryCheckpoint(CheckpointIO):
    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_dict = {}

    def load_checkpoint(self, path, map_location = None):
        return self.checkpoint_dict[path]
    
    def save_checkpoint(self, checkpoint, path, storage_options = None):
        self.checkpoint_dict[path] = deepcopy(checkpoint)

    def remove_checkpoint(self, path):
        del self.checkpoint_dict[path]
