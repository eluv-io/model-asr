from flask import Flask, jsonify, request
from marshmallow import Schema, fields
from dataclasses import dataclass, asdict
from dacite import from_dict
import argparse
from typing import List
from common_ml.tags import VideoTag
from flask_cors import CORS

from asr.model import EnglishSTT
from config import config

app = Flask(__name__)

@dataclass
class Args:
    word_level: bool

    class Schema(Schema):
        word_level = fields.Bool(missing=False)

@app.route('/info', methods=['GET'])
def info():
    return {"name": "Speech Recognition", "type": "audio"}

@app.route('/run', methods=['POST'])
def run():
    args = from_dict(data_class=Args, data=Args.Schema().load(request.args.to_dict()))
    files = request.files.getlist('files')
    print(files)
    model = EnglishSTT(config["asr_model"], config["lm_model"])
    all_tags = []
    for file in files:
        audio = file.read()
        tags = model.tag(audio)
        tags = prettify_tags(model, tags)
        if args.word_level:
            all_tags.append(tags)
        else:
            # combine into one tag
            all_tags.append([VideoTag(start_time=tags[0].start_time, end_time=tags[-1].end_time, text=' '.join(tag.text for tag in tags))])

    print(asdict(all_tags[0][0]))
    return jsonify({"result": [[asdict(tag) for tag in part_tags] for part_tags in all_tags]})

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
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    CORS(app)
    app.run(host='0.0.0.0', port=args.port)