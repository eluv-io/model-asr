from setuptools import setup
import os

setup(
    name="asr",
    version="0.1",
    packages=['asr'],
    install_requires=[
        'torch==1.9.0',
        'nemo_toolkit[all]==1.11.0',
        'torchaudio==0.9.0',
        'easydict==1.9',
        'Pillow==9.4.0',
        'scikit-learn==1.0.2',
        'wget',
        'psutil',
        'librosa',
        'tqdm',
        'sentencepiece==0.1.94',
        'argparse==1.4.0',
        'pycaption',
        'num2words==0.5.10',
        'word2number',
        'deepmultilingualpunctuation==1.0.1',
        'spacy==3.7.2',
        'loguru',
        'setproctitle',
        'common_ml @ git+ssh://git@github.com/qluvio/common-ml.git#egg=common_ml',
        'quick_test_py @ git+https://github.com/elv-nickB/quick_test_py.git#egg=quick_test_py'
    ]
)
