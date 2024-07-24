
from dataclasses import asdict
import argparse
from typing import List
from common_ml.tags import VideoTag
import json
import logging

from asr.model import EnglishSTT
from config import config

def run():
    files = args.audio_paths
    model = EnglishSTT(config["asr_model"], config["lm_model"])
    all_tags = []
    for file in files:
        with open(file, 'rb') as file:
            audio = file.read()
        tags = model.tag(audio)
        tags = prettify_tags(model, tags)
        if args.word_level:
            all_tags.append(tags)
        else:
            # combine into one tag
            all_tags.append([VideoTag(start_time=tags[0].start_time, end_time=tags[-1].end_time, text=' '.join(tag.text for tag in tags))])

    return {"result": [[asdict(tag) for tag in part_tags] for part_tags in all_tags]}

def prettify_tags(stt: EnglishSTT, asr_tags: List[VideoTag]) -> List[VideoTag]:
    if len(asr_tags) == 0:
        return asr_tags
    max_gap = 5000
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
    parser.add_argument('--word_level', action='store_true')
    args = parser.parse_args()
    logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)
    print(json.dumps(run()))