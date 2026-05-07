#!/bin/bash
# v2.1 ViT ffn_up dispatch tap experiment (single image, 1 decode token).
set -eo pipefail

BIN=~/AICAS2026/source/llama.cpp/build/bin/llama-mtmd-cli
MODEL=~/AICAS2026/AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf
MMPROJ=~/AICAS2026/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
IMAGE=~/AICAS2026/AICAS/image.png
LOG_DIR=~/AICAS2026/source/llama.cpp/logs/v2.1_vit_tap

rm -f /tmp/clip_tap.log /tmp/buft_tap.log /tmp/op_tap.log /tmp/exec_tap.log /tmp/assign_tap.log /tmp/status_tap.log

export GGML_XRT_XCLBIN=/lib/firmware/xilinx/fpga_gemm_v2.1/fpga_gemm_v2.1.xclbin
export GGML_XRT_CLIP_TAP=/tmp/clip_tap.log
export GGML_XRT_BUFT_TAP=/tmp/buft_tap.log
export GGML_XRT_OP_TAP=/tmp/op_tap.log
export GGML_XRT_EXEC_TAP=/tmp/exec_tap.log
export GGML_XRT_ASSIGN_TAP=/tmp/assign_tap.log
export GGML_XRT_STATUS_TAP=/tmp/status_tap.log

$BIN -m $MODEL --mmproj $MMPROJ --image $IMAGE -p "Describe the image." -n 1 2>&1 | tee $LOG_DIR/run.stderr.log
