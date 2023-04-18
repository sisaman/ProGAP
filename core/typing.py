from typing import Literal, TypeVar
from torch.types import Number


RT = TypeVar('RT')
Phase = Literal['train', 'val', 'test']
Metrics = dict[str, Number]
