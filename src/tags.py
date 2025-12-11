from dataclasses import dataclass

@dataclass
class VideoTag:
    start_time: int
    end_time: int
    text: str