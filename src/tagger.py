from typing import List
from dataclasses import dataclass
import os
import json
from dataclasses import asdict
import torch

from src.stt import EnglishSTT
from src.pretty import Prettifier
from src.tags import VideoTag
from src.audio import audio_file_to_tensor
from src.utils import combine_tags
from config import config


@dataclass
class RuntimeConfig:
    word_level: bool
    prettify: bool
    pretty_trail: bool
    pretty_trail_buffer: int


class AudioBuffer:
    """Accumulates audio tensors and metadata for trailing buffer processing"""
    
    def __init__(self):
        self.tensors: List[torch.Tensor] = []
        self.filenames: List[str] = []
        self.total_duration: float = 0.0
    
    def add(self, tensor: torch.Tensor, filename: str, duration: float):
        """Add audio to buffer"""
        self.tensors.append(tensor)
        self.filenames.append(filename)
        self.total_duration += duration
    
    def get_combined_tensor(self) -> torch.Tensor:
        """Concatenate all tensors"""
        return torch.cat(self.tensors, dim=1)
    
    def get_first_filename(self) -> str:
        """Get filename of first buffered audio"""
        return self.filenames[0] if self.filenames else ""
    
    def clear(self):
        """Clear the buffer"""
        self.tensors = []
        self.filenames = []
        self.total_duration = 0.0
    
    def is_ready(self, threshold: float) -> bool:
        """Check if buffer has reached threshold duration"""
        return self.total_duration >= threshold
    
    def is_empty(self) -> bool:
        return len(self.tensors) == 0


class SpeechTagger:
    """Orchestrates audio loading, STT tagging, prettification, and file writing"""
    
    def __init__(self, cfg: RuntimeConfig, tags_out: str):
        self.cfg = cfg
        self.tags_out = tags_out
        self.model = EnglishSTT(config["asr_model"], config["lm_model"])
        self.prettifier = Prettifier(config["postprocessing"]["sentence_gap"])
        
        # Initialize buffer for pretty_trail feature
        self.buffer = AudioBuffer() if cfg.pretty_trail else None
    
    def tag(self, fname: str) -> None:
        """
        Process a single audio file and write tag files
        
        Args:
            fname: Path to audio file
        """
        # Load audio file to tensor
        audio_tensor, duration = audio_file_to_tensor(fname)
        
        # Generate raw word-level tags
        tags = self.model.tag(audio_tensor)
        
        # Write primary output only if we have tags
        if len(tags) > 0:
            output_tags = self._format_tags(tags)
            self._write_tags(fname, output_tags, suffix="_tags.json")
        
        # Always add to trailing buffer if enabled (even if tags is empty)
        if self.cfg.pretty_trail:
            self._process_trailing_buffer(audio_tensor, fname, duration)
    
    def _format_tags(self, tags: List[VideoTag]) -> List[VideoTag]:
        """Apply prettification and word/phrase level formatting"""
        # Apply prettification if enabled
        if self.cfg.prettify:
            tags = self.prettifier.prettify(tags)
        
        # Convert to phrase-level if needed
        if not self.cfg.word_level:
            combined = combine_tags(tags)
            return [combined]
        
        return tags
    
    def _process_trailing_buffer(self, audio_tensor: torch.Tensor, fname: str, duration: float):
        """Handle accumulation and processing of trailing buffer"""
        # Add to buffer
        assert self.buffer is not None
        self.buffer.add(audio_tensor, fname, duration)
        
        # Check if buffer is ready to process
        if self.buffer.is_ready(self.cfg.pretty_trail_buffer):
            self._emit_prettified_trail()
    
    def _emit_prettified_trail(self):
        """Process accumulated buffer and emit prettified sentence-level tags"""
        assert self.buffer is not None
        if self.buffer.is_empty():
            return
        
        # Get combined audio
        combined_tensor = self.buffer.get_combined_tensor()
        first_fname = self.buffer.get_first_filename()
        
        # Run STT on combined audio
        tags = self.model.tag(combined_tensor)
        
        if len(tags) == 0:
            self.buffer.clear()
            return
        
        # Prettify word-level tags
        prettified_tags = self.prettifier.prettify(tags)
        
        # Merge into sentence-level tags
        sentence_tags = self._merge_to_sentences(prettified_tags)
        
        # Set source_media to first filename
        for tag in sentence_tags:
            tag.source_media = first_fname
        
        # Write output
        self._write_tags(first_fname, sentence_tags, suffix="_trail.json")
        
        # Clear buffer
        self.buffer.clear()
    
    def _merge_to_sentences(self, tags: List[VideoTag]) -> List[VideoTag]:
        """
        Merge word-level tags into sentence-level tags based on punctuation
        
        Args:
            tags: List of word-level tags (with punctuation from prettifier)
        
        Returns:
            List of sentence-level VideoTags
        """
        if len(tags) == 0:
            return []
        
        sentence_delimiters = {'.', '?', '!'}
        sentences = []
        current_words = []
        current_start = tags[0].start_time
        
        for i, tag in enumerate(tags):
            current_words.append(tag.text)
            
            # Check if this word ends with sentence delimiter
            if any(tag.text.endswith(delim) for delim in sentence_delimiters):
                # Create sentence tag
                sentence_tag = VideoTag(
                    start_time=current_start,
                    end_time=tag.end_time,
                    text=' '.join(current_words),
                    source_media=""  # Will be set by caller
                )
                sentences.append(sentence_tag)
                
                # Start new sentence
                current_words = []
                if i + 1 < len(tags):
                    current_start = tags[i + 1].start_time
        
        # Handle remaining words (if no sentence delimiter at end)
        if current_words:
            sentence_tag = VideoTag(
                start_time=current_start,
                end_time=tags[-1].end_time,
                text=' '.join(current_words),
                source_media=""
            )
            sentences.append(sentence_tag)
        
        return sentences
    
    def finalize(self):
        """Process any remaining buffered audio"""
        if self.cfg.pretty_trail and self.buffer and not self.buffer.is_empty():
            self._emit_prettified_trail()
    
    def _write_tags(self, fname: str, tags: List[VideoTag], suffix: str) -> None:
        """Write tags to JSON file"""
        output_path = os.path.join(
            self.tags_out, 
            f"{os.path.basename(fname)}{suffix}"
        )
        with open(output_path, 'w') as fout:
            fout.write(json.dumps([asdict(tag) for tag in tags]))