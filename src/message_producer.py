from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Dict
from dataclasses import dataclass

"""
All of this is copied from common_ml, but STT needs python 3.7 which is too old so we are not importing.
"""

@dataclass(frozen=True)
class Tag:
    start_time: int
    end_time: int
    tag: str
    source_media: str
    track: str = ""
    additional_info: Optional[Dict] = None

class Message: ...

@dataclass
class Progress:
    source_media: str

@dataclass
class Error:
    message: str
    source_media: Optional[str] = None

@dataclass
class TagMessage(Message):
    type: str
    data: Tag

@dataclass
class ProgressMessage(Message):
    type: str
    data: Progress

@dataclass
class ErrorMessage(Message):
    type: str
    data: Error

class TagMessageProducer(ABC):
    @abstractmethod
    def produce(self, files: List[str]) -> Iterator[Message]:
        pass