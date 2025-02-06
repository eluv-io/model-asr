#!/bin/bash

set -x

rm -rf test_output/
mkdir test_output

[ "$ELV_MODEL_TEST_GPU_TO_USE" != "" ] || ELV_MODEL_TEST_GPU_TO_USE=0

mkdir -p .cache

podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/test_output:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE asr test/1.mp4 test/2.mp4

ex=$?

jq . test_output/*.json

exit $ex
