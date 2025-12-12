from dataclasses import dataclass

@dataclass(frozen=True)
class ModelTag:
    start_time: int
    end_time: int
    text: str

@dataclass(frozen=True)
class AugmentedTag:
    start_time: int
    end_time: int
    text: str
    source_media: str
    track: str