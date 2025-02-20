
import argparse
from typing import List
import os
import json
from loguru import logger
from marshmallow import Schema, fields, ValidationError
from common_ml.tags import VideoTag
from dataclasses import asdict, dataclass
from common_ml.types import Data
from common_ml.utils import nested_update

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

def run(audio_paths: List[str], runtime_config: str=None) -> None:
    if runtime_config is None:
        cfg = config["runtime"]["default"]
    else:
        cfg = json.loads(runtime_config)
        cfg = nested_update(config["runtime"]["default"], cfg)
    try:
        runtime_config = RuntimeConfig.from_dict(cfg)
    except ValidationError as e:
        logger.error("Received invalid runtime config.")
        raise e
    tags_out = os.getenv('TAGS_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags'))
    if not os.path.exists(tags_out):
        os.makedirs(tags_out)
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
        with open(os.path.join(tags_out, f"{os.path.basename(fname).split('.')[0]}_tags.json"), 'w') as fout:
            fout.write(json.dumps([asdict(tag) for tag in tags]))

def prettify_tags(stt: EnglishSTT, asr_tags: List[VideoTag]) -> List[VideoTag]:
    if len(asr_tags) == 0:
        return asr_tags
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.audio_paths, args.config)