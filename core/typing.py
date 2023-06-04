from typing import Literal, TypeVar
from torch.types import Number


RT = TypeVar('RT')
Phase = Literal['train', 'val', 'test', 'predict']
Metrics = dict[str, Number]
