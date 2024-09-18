
from dataclasses import asdict
import argparse
from typing import List
import os
import json
from loguru import logger
from marshmallow import Schema, fields, ValidationError
from common_ml.tags import VideoTag

from asr.model import EnglishSTT
from config import config

class RuntimeConfig(Schema):
    word_level = fields.Bool(missing=False)

def run(audio_paths: List[str], runtime_config: str=None):
    files = audio_paths
    if runtime_config is None:
        cfg = config["runtime"]["default"]
    else:
        with open(runtime_config, 'r') as fin:
            cfg = json.load(fin)
    try:
        runtime_config = RuntimeConfig().load(cfg)
    except ValidationError as e:
        logger.error("Received invalid runtime config.")
        raise e
    tags_out = os.getenv('TAGS_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags'))
    if not os.path.exists(tags_out):
        os.makedirs(tags_out)
    model = EnglishSTT(config["asr_model"], config["lm_model"])
    for fname in files:
        with open(fname, 'rb') as fin:
            audio = fin.read()
        tags = model.tag(audio)
        tags = prettify_tags(model, tags)
        if not runtime_config['word_level']:
            # combine into one tag
            tags = [VideoTag(start_time=tags[0].start_time, end_time=tags[-1].end_time, text=' '.join(tag.text for tag in tags))]
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
    sentence_delimiters = ['.', '?', '!']
    last_tag = None
    for tag, corrected in zip(asr_tags, corrected_transcript.split(' ')):
        if last_tag and last_tag.start_time == tag.start_time and last_tag.text[-1] in sentence_delimiters:
            tag.start_time += 1
            tag.end_time += 1
        tag.text = corrected
        last_tag = tag
    return asr_tags

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.audio_paths, args.config)