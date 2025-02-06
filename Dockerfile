FROM continuumio/miniconda3:latest

WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential && apt-get install -y ffmpeg \
    && apt-get install -y curl

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
&& export PATH="$HOME/.cargo/bin:$PATH"

RUN \
   conda create -n asr python=3.7.16 -y

SHELL ["conda", "run", "-n", "asr", "/bin/bash", "-c"]

RUN \
    conda install -y cudatoolkit=10.1 cudnn=7 nccl

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

RUN \
    conda install -y rust

# Verify installation
RUN cargo --version

COPY setup.py .
RUN mkdir -p asr

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

RUN /opt/conda/envs/asr/bin/pip install .

COPY dependencies ./dependencies
RUN /opt/conda/envs/asr/bin/pip install dependencies/ctcdecode/.
RUN /opt/conda/envs/asr/bin/python -m spacy download en_core_web_sm

COPY models ./models

COPY config.yml run.py config.py .
COPY asr ./asr

ENTRYPOINT ["/opt/conda/envs/asr/bin/python", "run.py"]
