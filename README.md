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

#### Default run
`podman run --rm --network host --device nvidia.com/gpu=0 asr test/1.mp4 test/2.mp4`

#### Custom run

##### Option 1: change default runtime config
1. edit the `runtime/default` section in `config.yml`

##### Option 2: pass in custom runtime config as json file
2. `podman run --rm --network host --device nvidia.com/gpu=0 asr test/1.mp4 test/2.mp4 --config config.json`