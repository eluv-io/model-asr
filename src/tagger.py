from typing import List
from dataclasses import dataclass
import os
import json
from dataclasses import asdict

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


class SpeechTagger:
    """Orchestrates audio loading, STT tagging, prettification, and file writing"""
    
    def __init__(self, cfg: RuntimeConfig, tags_out: str):
        self.cfg = cfg
        self.tags_out = tags_out
        self.model = EnglishSTT(config["asr_model"], config["lm_model"])
        self.prettifier = Prettifier(config["postprocessing"]["sentence_gap"])
    
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
        
        if len(tags) == 0:
            return
        
        # Write primary output
        output_tags = self._format_tags(tags)
        self._write_tags(fname, output_tags, suffix="_tags.json")
        
        # Write secondary prettified track if enabled
        if self.cfg.pretty_trail:
            # TODO: implement trailing buffer logic
            pass
    
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
    
    def _write_tags(self, fname: str, tags: List[VideoTag], suffix: str) -> None:
        """Write tags to JSON file"""
        output_path = os.path.join(
            self.tags_out, 
            f"{os.path.basename(fname)}{suffix}"
        )
        with open(output_path, 'w') as fout:
            fout.write(json.dumps([asdict(tag) for tag in tags]))