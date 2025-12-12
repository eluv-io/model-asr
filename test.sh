#!/bin/bash

files() {
    echo test/1.mp4
    echo test/2.mp4
}

set -x

rm -rf test_output 
mkdir -p test_output/live test_output/reg

[ "$ELV_MODEL_TEST_GPU_TO_USE" != "" ] || ELV_MODEL_TEST_GPU_TO_USE=0

mkdir -p .cache

#files | podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/test_output/live:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE asr --live

podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/test_output/reg:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE asr test/1.mp4 test/2.mp4

ex=$?

jq . test_output/*/*.json

exit $ex
