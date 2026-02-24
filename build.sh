#!/bin/bash

set -e

git submodule update --init

SCRIPT_PATH="$(dirname "$(realpath "$0")")"

if [ ! -d "$SCRIPT_PATH/dependencides/ctcdecode/ctcdecode" ]; then
    ( cd "$SCRIPT_PATH"; git submodule update --init --recursive )
fi

MODEL_PATH=$(yq -r .storage.model_path $SCRIPT_PATH/config.yml)
mkdir -p models
rsync --progress --update --times --recursive --links --delete $MODEL_PATH/ $SCRIPT_PATH/models/stt/

exec buildscripts/build_container.bash -t "asr:${IMAGE_TAG:-latest}" -f Containerfile .