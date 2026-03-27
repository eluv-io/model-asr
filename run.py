
import argparse
import json
from dacite import from_dict
import setproctitle
from loguru import logger

from src.asr_producer import ASRProducer, RuntimeConfig
from src.default_loop import catch_errors, start_loop_from_producer, get_params

if __name__ == '__main__':
    catch_errors()
    setproctitle.setproctitle("model-asr")

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True)
    args, _ = parser.parse_known_args()

    params = get_params()
    params = from_dict(data=params, data_class=RuntimeConfig)
    
    producer = ASRProducer(params)
    start_loop_from_producer(producer, args.output_path, continue_on_error=True)