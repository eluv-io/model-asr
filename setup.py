from setuptools import setup

setup(
    name="asr",
    version="0.1",
    packages=['asr'],
    install_requires=[
        'torch==1.9.0',
        'nemo_toolkit[all]==1.11.0',
        'torchvision==0.10.0',
        'torchaudio==0.9.0',
        'scikit-image==0.17.2',
        'easydict==1.9',
        'Pillow==9.4.0',
        'scikit-learn==1.0.2',
        'pandas==1.3.5',
        'wget',
        'docopt',
        'schema',
        'psutil',
        'librosa',
        'tqdm',
        'nltk',
        'sentencepiece==0.1.94',
        'argparse==1.4.0',
        'facenet_pytorch==2.5.2',
        'mxnet-cu101',
        'pycaption',
        'num2words==0.5.10',
        'word2number',
        'deepmultilingualpunctuation==1.0.1',
        'spacy==3.7.2',
        'loguru',
        'dacite==1.8.1',
        'quick_test_py @ git+https://github.com/elv-nickB/quick_test_py.git#egg=quick_test_py',
        'common_ml @ git+https://github.com/elv-nickB/common-ml.git#egg=common_ml'
    ]
)