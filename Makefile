IMAGE_NAME := asr

DUMMY := $(shell git submodule update --init 1>&2)
include buildscripts/Makefile