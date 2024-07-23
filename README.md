# Setup

#### Download stt model
`python download_stt.py`

#### Pull submodules
`git submodule update --init --recursive`

#### Build image
`podman build --format docker -t asr . --network host`

#### Start container
`podman run asr --name asr -d --rm --network host -p 5000:5001`

#### Call API
`curl -X POST http://127.0.0.1:5001/run -F 'files=@test/1.mp4' -F 'files=@test/2.mp4'`