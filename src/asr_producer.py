from typing import List, Optional, Iterator
from dataclasses import dataclass, asdict
import torch
from loguru import logger

from src.stt import EnglishSTT
from src.pretty import Prettifier
from src.tags import ModelTag, AugmentedTag
from src.audio import audio_file_to_tensor
from src.utils import combine_tags
from config import config
from src.message_producer import (
    TagMessageProducer,
    Message,
    Tag,
    TagMessage,
    Progress,
    ProgressMessage,
    Error,
    ErrorMessage,
)


@dataclass(frozen=True)
class RuntimeConfig:
    word_level: bool = True
    prettify: bool = True
    pretty_trail: bool = True
    pretty_trail_buffer: int = 30


class AudioBuffer:
    """Accumulates audio tensors and metadata for trailing buffer processing"""

    def __init__(self):
        self.tensors: List[torch.Tensor] = []
        self.total_duration: float = 0.0

    def add(self, tensor: torch.Tensor, duration: float):
        self.tensors.append(tensor)
        self.total_duration += duration

    def get_combined_tensor(self) -> torch.Tensor:
        return torch.cat(self.tensors, dim=1)

    def clear(self):
        self.tensors = []
        self.total_duration = 0.0

    def is_ready(self, threshold: float) -> bool:
        return self.total_duration >= threshold

    def is_empty(self) -> bool:
        return len(self.tensors) == 0


class ASRProducer(TagMessageProducer):

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.model = EnglishSTT(config["asr_model"], config["lm_model"])
        self.prettifier = Prettifier(config["postprocessing"]["sentence_gap"])

    def produce(self, files: List[str]) -> Iterator[Message]:
        buffer = AudioBuffer() if self.cfg.pretty_trail else None
        pending_files: List[str] = []

        for fname in files:
            try:
                audio_tensor, duration = audio_file_to_tensor(fname)
                tags = self.model.tag(audio_tensor)

                if len(tags) > 0:
                    output_tags = self._format_tags(tags)
                    augmented = self._add_augmented_fields(output_tags, fname, None)
                    yield from self._tags_to_messages(augmented)

                if self.cfg.pretty_trail and buffer is not None:
                    buffer.add(audio_tensor, duration)
                    pending_files.append(fname)

                    if buffer.is_ready(self.cfg.pretty_trail_buffer):
                        yield from self._emit_prettified_trail(buffer, pending_files)
                        pending_files = []
                        buffer.clear()

            except Exception as e:
                logger.opt(exception=e).error(f"Error processing file {fname}")
                yield ErrorMessage(
                    type="error",
                    data=Error(source_media=fname, message=str(e)),
                )

        # Finalize: flush remaining buffer
        if self.cfg.pretty_trail and buffer is not None and not buffer.is_empty():
            yield from self._emit_prettified_trail(buffer, pending_files)

    def _format_tags(self, tags: List[ModelTag]) -> List[ModelTag]:
        if self.cfg.prettify:
            tags = self.prettifier.prettify(tags)
        if not self.cfg.word_level:
            combined = combine_tags(tags)
            return [combined]
        return tags

    def _add_augmented_fields(
        self, tags: List[ModelTag], fname: str, track: Optional[str]
    ) -> List[AugmentedTag]:
        return [
            AugmentedTag(
                start_time=tag.start_time,
                end_time=tag.end_time,
                tag=tag.tag,
                source_media=fname,
                track=track,
            )
            for tag in tags
        ]

    def _tags_to_messages(self, tags: List[AugmentedTag]) -> Iterator[TagMessage]:
        for tag in tags:
            data = asdict(tag)
            data = {k: v for k, v in data.items() if v is not None}
            yield TagMessage(
                type="tag",
                data=Tag(**data),
            )

    def _emit_prettified_trail(
        self, buffer: AudioBuffer, pending_files: List[str]
    ) -> Iterator[Message]:
        if buffer.is_empty() or len(pending_files) == 0:
            return

        combined_tensor = buffer.get_combined_tensor()
        first_fname = pending_files[0]

        tags = self.model.tag(combined_tensor)
        if len(tags) == 0:
            return

        prettified_tags = self.prettifier.prettify(tags)
        sentence_tags = self._merge_to_sentences(prettified_tags)
        augmented = self._add_augmented_fields(sentence_tags, first_fname, "auto_captions")

        yield from self._tags_to_messages(augmented)

        for fname in pending_files:
            yield ProgressMessage(
                type="progress",
                data=Progress(source_media=fname),
            )

    def _merge_to_sentences(self, tags: List[ModelTag]) -> List[ModelTag]:
        if len(tags) == 0:
            return []

        sentence_delimiters = {'.', '?', '!'}
        sentences = []
        current_words = []
        current_start = tags[0].start_time

        for i, tag in enumerate(tags):
            current_words.append(tag.tag)

            if any(tag.tag.endswith(delim) for delim in sentence_delimiters):
                sentences.append(ModelTag(
                    start_time=current_start,
                    end_time=tag.end_time,
                    tag=' '.join(current_words),
                ))
                current_words = []
                if i + 1 < len(tags):
                    current_start = tags[i + 1].start_time

        if current_words:
            sentences.append(ModelTag(
                start_time=current_start,
                end_time=tags[-1].end_time,
                tag=' '.join(current_words),
            ))

        return sentences
