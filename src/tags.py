from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ModelTag:
    start_time: int
    end_time: int
    tag: str

@dataclass(frozen=True)
class AugmentedTag:
    start_time: int
    end_time: int
    tag: str
    source_media: str
    track: Optional[str]