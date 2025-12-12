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
        'wget==3.2',
        'psutil==7.0.0',
        'librosa==0.10.2.post1',
        'tqdm==4.67.1',
        'sentencepiece==0.1.94',
        'argparse==1.4.0',
        'pycaption==2.2.1',
        'num2words==0.5.10',
        'word2number==1.1',
        'deepmultilingualpunctuation==1.0.1',
        'spacy==3.7.2',
        'loguru==0.7.3',
        'setproctitle==1.3.3',
        'common_ml @ git+ssh://git@github.com/eluv-io/common-ml.git@3d3bf2ce2fa8b6eb84e830650569f53a979becc8#egg=common_ml',
    ]
)
