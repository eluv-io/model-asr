
import argparse
from typing import List, Callable
import os
import sys
import json
from dacite import from_dict
import setproctitle
import time
from queue import Queue
import threading

from src.tagger import SpeechTagger, RuntimeConfig
from src.utils import nested_update
from config import config

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

def make_tag_fn(cfg: RuntimeConfig, tags_out: str) -> Callable:
    """
    Create a function that processes audio files using SpeechTagger
    
    Args:
        cfg: Runtime configuration
        
    Returns:
        Function that takes list of audio file paths
    """
    tagger = SpeechTagger(cfg, tags_out)
    
    def tag_fn(audio_paths: List[str]) -> None:
        for fname in audio_paths:
            try:
                tagger.tag(fname)
            except Exception as e:
                print(f"Error processing {fname}: {e}", file=sys.stderr)
    
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
    
    tag_fn = make_tag_fn(runtime_config, tags_out)

    if args.live:
        print('Running in live mode', file=sys.stderr)
        run_live_mode(tag_fn)
    else:
        tag_fn(args.audio_paths)