FROM continuumio/miniconda3:latest

WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential && \
    apt-get install -y ffmpeg

RUN \
   conda create -n asr python=3.7.16 -y

SHELL ["conda", "run", "-n", "asr", "/bin/bash", "-c"]

RUN \
    conda install -y cudatoolkit=10.1 cudnn=7 nccl && \
    conda install -y -c conda-forge ffmpeg-python

COPY . .

RUN /opt/conda/envs/asr/bin/pip install .
RUN /opt/conda/envs/asr/bin/pip install dependencies/ctcdecode/.
RUN /opt/conda/envs/asr/bin/python -m spacy download en_core_web_sm

ENTRYPOINT ["/opt/conda/envs/asr/bin/python", "run.py"]