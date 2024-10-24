from typing import List, Callable
import json
import os
import argparse
from quick_test_py import Tester

from run import run

def test_stt() -> List[Callable]:
    filedir = os.path.dirname(os.path.abspath(__file__))
    def test():
        os.environ['TAGS_PATH'] = os.path.join(filedir, 'tags')
        run([os.path.join(filedir, '1.mp4'), os.path.join(filedir, '2.mp4')])
        tags = []
        for tag_file in sorted(os.listdir(os.environ['TAGS_PATH'])):
            with open(os.path.join(os.environ['TAGS_PATH'], tag_file), 'r') as fin:
                tags.append(json.load(fin))
        return tags
    return [test]

def main():
    filedir = os.path.dirname(os.path.abspath(__file__))
    tester = Tester(os.path.join(filedir, 'test_data'))
    tester.register('test_stt', test_stt())
    if args.record:
        tester.record()
    else:
        tester.validate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    main()