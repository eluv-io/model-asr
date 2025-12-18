FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN conda create -n mlpod python=3.7.16 -y

RUN apt-get update && apt-get install -y build-essential && apt-get install -y ffmpeg

RUN conda run -n mlpod conda install -y cudatoolkit=10.1 cudnn=7 nccl
RUN conda run -n mlpod conda install -y rust

# Verify installation of rust
RUN cargo --version

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

COPY models ./models

COPY setup.py .

RUN /opt/conda/envs/mlpod/bin/pip install .

COPY dependencies ./dependencies
RUN /opt/conda/envs/mlpod/bin/pip install dependencies/ctcdecode/.

COPY config.yml run.py config.py .
RUN mkdir -p src
COPY src ./src

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "-u", "run.py"]
