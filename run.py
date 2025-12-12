
import argparse
import sys
from typing import List, Callable
import os
import json
from loguru import logger
from marshmallow import Schema, fields, ValidationError
from common_ml.tags import VideoTag
from dataclasses import asdict, dataclass
from common_ml.types import Data
from common_ml.utils import nested_update
from not_common_ml.model import run_live_mode
import setproctitle

from asr.model import EnglishSTT
from config import config

@dataclass
class RuntimeConfig(Data):
    word_level: bool
    class Schema(Schema):
        word_level = fields.Bool(required=True)

    @staticmethod
    def from_dict(data: dict) -> 'RuntimeConfig':
        return RuntimeConfig(**data)

def run(audio_paths: List[str], runtime_config: dict = {}, tags_out: str = ".") -> None:

    model = EnglishSTT(config["asr_model"], config["lm_model"])

    word_tags = []
    word_tags_by_file = []
    for fname in audio_paths:
        with open(fname, 'rb') as fin:
            audio = fin.read()
        tags = model.tag(audio)
        word_tags.extend(tags)
        word_tags_by_file.append(tags)

    transcript = prettify_tags(model, word_tags)
    transcript = transcript.split(' ')

    idx = 0
    for fname, tags in zip(audio_paths, word_tags_by_file):
        for tag in tags:
            tag.text = transcript[idx]
            idx += 1
        with open(os.path.join(tags_out, f"{os.path.basename(fname)}_tags.json"), 'w') as fout:
            fout.write(json.dumps([asdict(tag) for tag in tags]))

def prettify_tags(stt: EnglishSTT, asr_tags: List[VideoTag]) -> str:
    if len(asr_tags) == 0:
        return ""
    max_gap = config["postprocessing"]["sentence_gap"]
    full_transcript = [asr_tags[0].text]
    last_start = asr_tags[0].start_time
    for tag in asr_tags[1:]:
        if tag.start_time - last_start > max_gap:
            full_transcript.append(tag.text)
        else:
            full_transcript[-1] += ' ' + tag.text
        last_start = tag.start_time
    corrected_transcript = [stt.correct_text(t) for t in full_transcript]
    corrected_transcript = ' '.join(corrected_transcript)
    return corrected_transcript

def make_tag_fn(runtime_config: dict = {}, tags_out: str = ".") -> Callable:

    model = EnglishSTT(config["asr_model"], config["lm_model"])

    def tag_fn(audio_paths: List[str]) -> None:

        word_tags = []
        word_tags_by_file = []
        for fname in audio_paths:
            with open(fname, 'rb') as fin:
                audio = fin.read()
            tags = model.tag(audio)
            word_tags.extend(tags)
            word_tags_by_file.append(tags)

        transcript = prettify_tags(model, word_tags)
        transcript = transcript.split(' ')

        idx = 0
        for fname, tags in zip(audio_paths, word_tags_by_file):
            for tag in tags:
                tag.text = transcript[idx]
                idx += 1
            with open(os.path.join(tags_out, f"{os.path.basename(fname)}_tags.json"), 'w') as fout:
                fout.write(json.dumps([asdict(tag) for tag in tags]))

    return tag_fn

if __name__ == '__main__':
    setproctitle.setproctitle("model-asr")
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_paths', nargs='*', type=str)
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--live', action='store_true', help='Run in live mode (read files from stdin)')
    args = parser.parse_args()

    if args.config is None:
        cfg = config["runtime"]["default"]
    else:
        cfg = json.loads(args.config)
        cfg = nested_update(config["runtime"]["default"], cfg)
    try:
        runtime_config = RuntimeConfig.from_dict(cfg)
    except ValidationError as e:
        logger.error("Received invalid runtime config.")
        raise e
    
    tags_out = os.getenv('TAGS_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags'))

    if not os.path.exists(tags_out):
        os.makedirs(tags_out)

    if not args.live and len(args.audio_paths) == 0:
        logger.error("No files to tag, and not live mode")
        raise ValidationError("invalid args")
    
    tag_fn = make_tag_fn(runtime_config, tags_out)

    if args.live:
        print('Running in live mode', file=sys.stderr)
        run_live_mode(tag_fn)
    else:
        tag_fn(args.audio_paths)
