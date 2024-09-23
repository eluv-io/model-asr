# Setup

## With Podman

### Dependencies
1. Podman with nvidia toolkit enabled
2. Python

#### Download stt model
`python download_weights.py`

#### Pull submodules
`git submodule update --init --recursive`

#### Build image
`podman build --format docker -t asr . --network host`

#### Default run
`podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/models:/elv/models:ro --volume=$(pwd)/tags:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=0 caption test/1.mp4 test/2.mp4`

1. Note: you must mount the files to tag into the container storage (`--volume=$(pwd)/test:/elv/test`)
2. Tag files will appear in the `tags` directory (`--volume=$(pwd)/tags:/elv/tags`). 

#### Custom run

##### Option 1: change default runtime config
1. edit the `runtime/default` section in `config.yml`

##### Option 2: pass in custom runtime config as json file
2. `podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/models:/elv/models:ro --volume=$(pwd)/tags:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=0 asr test/1.mp4 test/2.mp4 --config config.json`

## Local testing

1. Set up on local system: follow steps in docker file. 
2. run `python test/test.py`


