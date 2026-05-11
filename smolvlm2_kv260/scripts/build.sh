#!/bin/bash
# =============================================================================
# build.sh  —  SmolVLM2 KV260 完整构建流程
# =============================================================================
# 阶段：
#   1. HLS 综合（vitis_hls）→ .xo
#   2. v++ 编译（如有多个 kernel，单 kernel 跳过此步）
#   3. v++ 链接（v++ -l）→ .xclbin
#   4. 可选：打包 sd_card 镜像
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# 环境变量（根据实际 Vitis 安装路径修改）
# ---------------------------------------------------------------------------
export XILINX_VITIS=${XILINX_VITIS:-/tools/Xilinx/Vitis/2023.2}
export XILINX_XRT=${XILINX_XRT:-/opt/xilinx/xrt}
export PLATFORM=${PLATFORM:-xilinx_kv260_smartcamera_202320_1}

source ${XILINX_VITIS}/settings64.sh
source ${XILINX_XRT}/setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/build"
HLS_DIR="${ROOT_DIR}/scripts"
CFG_DIR="${ROOT_DIR}/cfg"

mkdir -p "${BUILD_DIR}"

ENABLE_MULTI_KERNEL=${ENABLE_MULTI_KERNEL:-0}

run_hls_if_missing() {
    local tcl_name="$1"
    local xo_name="$2"
    if [ ! -f "${HLS_DIR}/${xo_name}" ]; then
        vitis_hls -f "${tcl_name}" 2>&1 | tee "${BUILD_DIR}/${tcl_name%.tcl}.log"
    else
        echo "[BUILD] ${xo_name} already exists, skipping ${tcl_name}."
    fi
}

# ---------------------------------------------------------------------------
# 步骤 1：HLS 综合（生成 .xo）
# ---------------------------------------------------------------------------
echo "========================================"
echo "  STEP 1: HLS Synthesis"
echo "========================================"
cd "${HLS_DIR}"

run_hls_if_missing hls_synth.tcl smolvlm2_decoder_kernel.xo
cp smolvlm2_decoder_kernel.xo "${BUILD_DIR}/"

if [ "${ENABLE_MULTI_KERNEL}" = "1" ]; then
    run_hls_if_missing hls_synth_vit.tcl smolvlm2_vit_kernel.xo
    run_hls_if_missing hls_synth_connector.tcl smolvlm2_connector_kernel.xo
    run_hls_if_missing hls_synth_bridge.tcl smolvlm2_bridge_kernel.xo
    cp smolvlm2_vit_kernel.xo "${BUILD_DIR}/"
    cp smolvlm2_connector_kernel.xo "${BUILD_DIR}/"
    cp smolvlm2_bridge_kernel.xo "${BUILD_DIR}/"
    echo "[BUILD] Multi-kernel overlay enabled."
    echo "        XO files: decoder + vit_prefill/generic_gemm + connector + bridge"
else
    echo "[BUILD] ENABLE_MULTI_KERNEL=0, only decoder XO will be linked."
fi

cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 步骤 2：v++ 链接（HW 综合，时间较长：30~90 分钟）
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  STEP 2: v++ Link (HW Synthesis)"
echo "========================================"

XO_FILE="${HLS_DIR}/smolvlm2_decoder_kernel.xo"
XCLBIN="${BUILD_DIR}/smolvlm2_decoder.xclbin"
LINK_CFG="${CFG_DIR}/kv260_link.cfg"
XO_FILES=("${XO_FILE}")

if [ "${ENABLE_MULTI_KERNEL}" = "1" ]; then
    LINK_CFG="${CFG_DIR}/kv260_link_multi.cfg"
    XO_FILES+=("${HLS_DIR}/smolvlm2_vit_kernel.xo")
    XO_FILES+=("${HLS_DIR}/smolvlm2_connector_kernel.xo")
    XO_FILES+=("${HLS_DIR}/smolvlm2_bridge_kernel.xo")
fi

if [ ! -f "${XCLBIN}" ]; then
    v++ \
        --link \
        --target hw \
        --platform ${PLATFORM} \
        --config "${LINK_CFG}" \
        --save-temps \
        --temp_dir "${BUILD_DIR}/vpp_temp" \
        --log_dir  "${BUILD_DIR}/vpp_logs" \
        --report_dir "${BUILD_DIR}/vpp_reports" \
        -o "${XCLBIN}" \
        "${XO_FILES[@]}" \
        2>&1 | tee "${BUILD_DIR}/vpp_link.log"
    echo "[BUILD] v++ link done: ${XCLBIN}"
else
    echo "[BUILD] XCLBIN already exists: ${XCLBIN}"
fi

# ---------------------------------------------------------------------------
# 步骤 3：检查时序报告（WNS 必须 > 0）
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  STEP 3: Timing Check"
echo "========================================"

TIMING_RPT=$(find "${BUILD_DIR}/vpp_temp" -name "*.wns" 2>/dev/null | head -1)
if [ -n "${TIMING_RPT}" ]; then
    echo "[BUILD] Timing report: ${TIMING_RPT}"
    cat "${TIMING_RPT}"
fi

# 从综合报告提取 WNS
WNS_FILE=$(find "${BUILD_DIR}/vpp_temp" -name "*.rpt" | xargs grep -l "WNS" 2>/dev/null | head -1)
if [ -n "${WNS_FILE}" ]; then
    echo "[BUILD] WNS summary:"
    grep "WNS" "${WNS_FILE}" | head -5
fi

echo ""
echo "========================================"
echo "  BUILD COMPLETE"
echo "  XCLBIN: ${XCLBIN}"
echo "========================================"

# ---------------------------------------------------------------------------
# 步骤 4（可选）：生成 SD 卡镜像
# ---------------------------------------------------------------------------
# 如需打包 KV260 boot 镜像，取消下面注释
# echo ""
# echo "Packaging SD card image..."
# v++ --package \
#     --platform ${PLATFORM} \
#     --target hw \
#     "${XCLBIN}" \
#     --package.out_dir "${BUILD_DIR}/sd_card" \
#     --package.rootfs  <path_to_rootfs.ext4> \
#     --package.image   <path_to_boot.scr>
