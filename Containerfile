# dependencies are impossible - using old base image to build correctly
FROM us-docker.pkg.dev/github-qluvio/ml/asr:legacy-live
WORKDIR /elv

RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

COPY config.yml run.py config.py .
COPY src ./src

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "-u", "run.py"]
