#!/bin/bash
set -u
BIN=~/AICAS2026/source/llama.cpp/build/bin/llama-mtmd-cli
MODEL=~/AICAS2026/AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf
MMPROJ=~/AICAS2026/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
IMAGE=~/AICAS2026/AICAS/image.png
LOG=$1
shift
"$@" $BIN -m $MODEL --mmproj $MMPROJ --image $IMAGE -p 'Describe the image.' -n 1 > $LOG 2>&1
echo exit=$?
grep -E 'image slice encoded|image decoded|prompt eval time|total time|llama_perf_context_print:.*eval time' $LOG | head -10
