
import argparse
from typing import List, Callable
import os
import sys
import json
from dataclasses import asdict, dataclass
from dacite import from_dict
import setproctitle
import time
from queue import Queue
import threading

from src.stt import EnglishSTT
from src.utils import nested_update
from src.tags import VideoTag
from config import config

@dataclass
class RuntimeConfig:
    word_level: bool
    prettify: bool

def run_live_mode(
    tag_fn: Callable[[List[str]], None], 
    batch_timeout: float=0.2,
) -> None:
    """
    Live mode: reads file paths from stdin and processes them in batches
    
    Args:
        tag_fn: Function that takes (file_paths: List[str])
    """
    
    file_queue = Queue()
    
    def stdin_reader():
        """Thread function to read from stdin and add files to queue"""
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    file_queue.put(line)
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            file_queue.put(None)  # Signal end of input
    
    def process_batch(files):
        """Process a batch of files using the provided function"""
        valid_files = []
        for f in files:
            if os.path.exists(f):
                valid_files.append(f)
            else:
                print(f"Warning: file {f} does not exist, skipping", file=sys.stderr)
        if valid_files:
            print(f"Processing batch of {len(valid_files)} files...", file=sys.stderr)
            tag_fn(valid_files)
            print(f"Completed batch of {len(valid_files)} files", file=sys.stderr)
    
    reader_thread = threading.Thread(target=stdin_reader, daemon=True)
    reader_thread.start()
    
    current_batch = []
    
    while True:
        try:
            while not file_queue.empty():
                try:
                    file_path = file_queue.get_nowait()
                    
                    if file_path is None:
                        if current_batch:
                            process_batch(current_batch)
                        return
                    
                    current_batch.append(file_path)
                except:
                    break
            
            if current_batch:
                process_batch(current_batch)
                current_batch = []
            
            if not reader_thread.is_alive() and file_queue.empty():
                break
            
            time.sleep(batch_timeout)
                
        except KeyboardInterrupt:
            break

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

def make_tag_fn(cfg : RuntimeConfig) -> Callable:

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

        if cfg.prettify:
            transcript = prettify_tags(model, word_tags)
        else:
            transcript = " ".join(t.text for t in word_tags)

        transcript = transcript.split(' ')

        idx = 0
        for fname, tags in zip(audio_paths, word_tags_by_file):
            if len(tags) == 0:
                continue

            out_tags = []

            if cfg.word_level:
                for tag in tags:
                    tag.text = transcript[idx]
                    out_tags.append(tag)
                    idx += 1
            else:
                new_tag = VideoTag(
                    start_time=tags[0].start_time, 
                    end_time=tags[-1].end_time,
                    text = " ".join(transcript[idx:idx+len(tags)])
                )
                idx += len(tags)
                out_tags.append(new_tag)

            with open(os.path.join(tags_out, f"{os.path.basename(fname)}_tags.json"), 'w') as fout:
                fout.write(json.dumps([asdict(tag) for tag in out_tags]))

    return tag_fn

if __name__ == '__main__':
    setproctitle.setproctitle("model-asr")
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_paths', nargs='*', type=str, default=[])
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--live', action='store_true', help='Run in live mode (read files from stdin)')
    args = parser.parse_args()
    
    if args.config is None:
        cfg = config["runtime"]["default"]
    else:
        cfg = json.loads(args.config)
        cfg = nested_update(config["runtime"]["default"], cfg)

    runtime_config = from_dict(data=cfg, data_class=RuntimeConfig)

    tags_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    if not os.path.exists(tags_out):
        os.makedirs(tags_out)
    
    tag_fn = make_tag_fn(runtime_config)

    if args.live:
        print('Running in live mode', file=sys.stderr)
        run_live_mode(tag_fn)
    else:
        tag_fn(args.audio_paths)