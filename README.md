# Setup

#### Dependencies
1. Python
2. Podman with nvidia toolkit enabled

#### Download stt model
`python download_stt.py`

#### Pull submodules
`git submodule update --init --recursive`

#### Build image
`podman build --format docker -t asr . --network host`

#### Run container
`podman run --rm --network host --device nvidia.com/gpu=0 asr test/1.mp4 test/2.mp4`