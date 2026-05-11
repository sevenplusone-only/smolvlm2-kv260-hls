# =============================================================================
# Makefile  —  SmolVLM2 KV260 工程构建
# =============================================================================
# 目标：
#   make csim      : HLS C 仿真（快速功能验证，~5分钟）
#   make synth     : HLS 综合（生成 .xo，~15分钟）
#   make link      : v++ 链接（生成 .xclbin，~60分钟）
#   make vit_synth : HLS 综合 ViT kernel
#   make conn_synth: HLS 综合 connector kernel
#   make host      : 编译 PS 端 host_driver
#   make quant     : 运行离线权重量化
#   make check_dims: 检查 config.json 与工程目标尺寸
#   make clean     : 清理构建产物
# =============================================================================

# ---------------------------------------------------------------------------
# 工具路径（根据实际安装修改）
# ---------------------------------------------------------------------------
VITIS_HLS ?= vitis_hls
VPP       ?= v++
PYTHON    ?= python3
GXX       ?= g++
XRT_INC   ?= /opt/xilinx/xrt/include
XRT_LIB   ?= /opt/xilinx/xrt/lib

# ---------------------------------------------------------------------------
# 源文件路径
# ---------------------------------------------------------------------------
SRC_DIR    := src
TB_DIR     := tb
SCRIPT_DIR := scripts
CFG_DIR    := cfg
BUILD_DIR  := build

# ---------------------------------------------------------------------------
# .PHONY 目标
# ---------------------------------------------------------------------------
.PHONY: all csim synth vit_synth conn_synth bridge_synth link link_multi host quant quant_exp check_dims clean check_env

all: synth

# ---------------------------------------------------------------------------
# 检查环境
# ---------------------------------------------------------------------------
check_env:
	@which $(VITIS_HLS) > /dev/null 2>&1 || \
	    (echo "ERROR: vitis_hls not found. Source Vitis settings64.sh." && exit 1)
	@echo "[ENV] vitis_hls: $$(which $(VITIS_HLS))"
	@echo "[ENV] Platform: $${PLATFORM:-xilinx_kv260_smartcamera_202320_1}"

# ---------------------------------------------------------------------------
# csim：C 仿真（最快验证，不需要 License）
# ---------------------------------------------------------------------------
csim: check_env
	@echo "========================================"
	@echo "  C Simulation (csim)"
	@echo "========================================"
	@mkdir -p $(BUILD_DIR)
	cd $(SCRIPT_DIR) && $(VITIS_HLS) -f hls_synth.tcl 2>&1 | tee ../$(BUILD_DIR)/csim.log
	@grep -E "PASS|FAIL|ERROR|error" $(BUILD_DIR)/csim.log | head -20 || true
	@echo ""
	@echo "csim log: $(BUILD_DIR)/csim.log"

# ---------------------------------------------------------------------------
# synth：HLS 综合（生成 RTL + .xo）
# ---------------------------------------------------------------------------
synth: check_env
	@echo "========================================"
	@echo "  HLS Synthesis"
	@echo "========================================"
	@mkdir -p $(BUILD_DIR)
	cd $(SCRIPT_DIR) && $(VITIS_HLS) -f hls_synth.tcl 2>&1 | tee ../$(BUILD_DIR)/hls.log
	@echo ""
	@echo "[SYNTH] Checking resource usage..."
	@grep -E "DSP|BRAM|URAM|LUT|Timing|WNS" $(BUILD_DIR)/hls.log | head -20 || true

vit_synth: check_env
	@echo "========================================"
	@echo "  HLS Synthesis (ViT)"
	@echo "========================================"
	@mkdir -p $(BUILD_DIR)
	cd $(SCRIPT_DIR) && $(VITIS_HLS) -f hls_synth_vit.tcl 2>&1 | tee ../$(BUILD_DIR)/hls_vit.log

conn_synth: check_env
	@echo "========================================"
	@echo "  HLS Synthesis (Connector)"
	@echo "========================================"
	@mkdir -p $(BUILD_DIR)
	cd $(SCRIPT_DIR) && $(VITIS_HLS) -f hls_synth_connector.tcl 2>&1 | tee ../$(BUILD_DIR)/hls_connector.log

bridge_synth: check_env
	@echo "========================================"
	@echo "  HLS Synthesis (Image->Decoder Bridge)"
	@echo "========================================"
	@mkdir -p $(BUILD_DIR)
	cd $(SCRIPT_DIR) && $(VITIS_HLS) -f hls_synth_bridge.tcl 2>&1 | tee ../$(BUILD_DIR)/hls_bridge.log

# ---------------------------------------------------------------------------
# link：v++ 链接（时间最长，需要 Vivado License）
# ---------------------------------------------------------------------------
link: check_env
	@echo "========================================"
	@echo "  v++ Link (Vivado Implementation)"
	@echo "========================================"
	bash $(SCRIPT_DIR)/build.sh

link_multi: check_env
	@echo "========================================"
	@echo "  v++ Link (Decoder + ViT + Connector)"
	@echo "========================================"
	ENABLE_MULTI_KERNEL=1 bash $(SCRIPT_DIR)/build.sh

# ---------------------------------------------------------------------------
# host：编译 PS 端驱动（在 KV260 ARM 上编译，或交叉编译）
# ---------------------------------------------------------------------------
host:
	@echo "========================================"
	@echo "  Compile Host Driver"
	@echo "========================================"
	$(GXX) -O2 -std=c++17 \
	    -I$(XRT_INC) \
	    $(SCRIPT_DIR)/host_driver.cpp \
	    -o $(BUILD_DIR)/smolvlm2_infer \
	    -L$(XRT_LIB) -lxrt_coreutil -lxrt_core \
	    -Wl,-rpath,$(XRT_LIB)
	@echo "[HOST] Binary: $(BUILD_DIR)/smolvlm2_infer"

# ---------------------------------------------------------------------------
# quant：运行离线量化
# ---------------------------------------------------------------------------
quant:
	@echo "========================================"
	@echo "  Weight Quantization (W4 RTN)"
	@echo "========================================"
	$(PYTHON) $(SCRIPT_DIR)/quantize_weights.py
	@echo "[QUANT] Done. Check quantized_weights/"

quant_exp:
	@echo "========================================"
	@echo "  Unified Quant Experiment Pipeline"
	@echo "========================================"
	$(PYTHON) $(SCRIPT_DIR)/run_quant_experiment.py --scheme all --stage all --output-dir quant_experiments
	@echo "[QUANT_EXP] Done. Check quant_experiments/"

check_dims:
	$(PYTHON) $(SCRIPT_DIR)/check_dimensions.py

# ---------------------------------------------------------------------------
# 独立 C++ 编译（用于快速 csim debug，不需要 Vitis HLS）
# ---------------------------------------------------------------------------
CFLAGS := -I$(SRC_DIR) -O2 -std=c++14
# ap_int/ap_fixed 头文件路径（Vitis HLS 安装目录）
HLS_INC ?= $(shell $(VITIS_HLS) -e include 2>/dev/null || echo "/tools/Xilinx/Vitis_HLS/2023.2/include")

native_csim:
	@echo "========================================"
	@echo "  Native C++ Compile (for quick debug)"
	@echo "========================================"
	$(GXX) $(CFLAGS) -I$(HLS_INC) \
	    $(TB_DIR)/decoder_tb.cpp \
	    $(SRC_DIR)/decoder_kernel.cpp \
	    -o $(BUILD_DIR)/decoder_tb_native \
	    -lm
	./$(BUILD_DIR)/decoder_tb_native

# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(SCRIPT_DIR)/smolvlm2_decoder
	rm -f  $(SCRIPT_DIR)/smolvlm2_decoder_kernel.xo
	rm -f  vitis_hls.log
	@echo "Clean done."

# ---------------------------------------------------------------------------
# 资源估算（基于文档数据，综合前快速检查）
# ---------------------------------------------------------------------------
estimate:
	@echo "=== 资源估算（xck26，参考文档数据）==="
	@echo "DSP:  TM×TN×K_UNROLL/2 × tiles = (2×2×32)/2 × 8tiles = 512 (预算 998)"
	@echo "BRAM: QT×KVT×2B×2bank = 32×64×2×2 = 8KB → 2 BRAM36K (预算 115)"
	@echo "URAM: act_buf(512×960×1B) = 480KB → 2 URAM288K (预算 51)"
	@echo "      kv_buf(2bank×64seqmax×64dim×0.5B) = 4KB → shared in BRAM"
	@echo "LUT:  rmsnorm(sqrt+div) + rope_lut + exp_lut + silu_lut 约 20K (预算 94K)"
	@echo ""
	@echo "带宽需求（Decode，seqlen=1）："
	@echo "  权重: 5 gemm × 960×N × 0.5B / 0.5ms ≈ 8 GB/s (满足)"
	@echo "  KV:   5 heads × 2000seq × 64dim × 0.5B / 0.5ms ≈ 0.6 GB/s (轻松)"
