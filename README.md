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

#### Start container
`podman run -d --rm --name asr_server --network host --device nvidia.com/gpu=0 asr`

#### Call API
`curl -X POST http://127.0.0.1:5001/run -F 'files=@test/1.mp4' -F 'files=@test/2.mp4'`