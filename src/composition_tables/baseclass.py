from abc import ABC
from dataclasses import dataclass
from typing import List

@dataclass
class SemiGroup(ABC):
    elements: List[int]

    