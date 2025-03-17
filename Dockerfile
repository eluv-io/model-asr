FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential && apt-get install -y ffmpeg

RUN conda create -n asr python=3.7.16 -y

SHELL ["conda", "run", "-n", "tagenv", "/bin/bash", "-c"]

RUN conda install -y cudatoolkit=10.1 cudnn=7 nccl

RUN conda install -y rust

# Verify installation of rust
RUN cargo --version

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

COPY models ./models

COPY setup.py .
RUN mkdir -p asr

RUN /opt/conda/envs/tagenv/bin/pip install .

COPY dependencies ./dependencies
RUN /opt/conda/envs/tagenv/bin/pip install dependencies/ctcdecode/.
RUN /opt/conda/envs/tagenv/bin/python -m spacy download en_core_web_sm

COPY config.yml run.py config.py .
COPY asr ./asr

ENTRYPOINT ["/opt/conda/envs/tagenv/bin/python", "run.py"]
