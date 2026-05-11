#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_DIR="${1:-${ROOT_DIR}/quant_experiments_awq_repair_20260508_010000/w4a8_awq_g32/hardware_export}"
MANIFEST_NAME="${2:-hardware_export_manifest.repaired.json}"

echo "[1/6] strict hardware export check"
python3 "${ROOT_DIR}/scripts/check_hardware_export.py" "${EXPORT_DIR}" --manifest-name "${MANIFEST_NAME}"

echo
echo "[2/6] transfer budget"
python3 "${ROOT_DIR}/scripts/analyze_transfer_budget.py" "${EXPORT_DIR}" --manifest-name "${MANIFEST_NAME}" \
  --csv-out /private/tmp/smolvlm2_transfer_budget.csv \
  --json-out /private/tmp/smolvlm2_transfer_budget.json

echo
echo "[2b/6] config/hardware seq cap"
python3 - <<'PY'
import json
cfg=json.load(open('/Users/miracle/smolvlm_local/config.json','r',encoding='utf-8'))
model_max=cfg['text_config']['max_position_embeddings']
hw_max=2048
mode='restricted' if model_max != hw_max else 'native'
print(f"model_max_seq = {model_max}")
print(f"hw_max_seq = {hw_max}")
print(f"seq_cap_mode = {mode}")
PY

echo
echo "[3/6] feasibility estimate"
python3 "${ROOT_DIR}/scripts/estimate_kv260_feasibility.py" "${EXPORT_DIR}" --manifest-name "${MANIFEST_NAME}"

echo
echo "[4/6] static dataflow audit"
python3 "${ROOT_DIR}/scripts/audit_dataflow_static.py"

echo
echo "[5/6] C++ structure check with local XRT stubs"
STUB_DIR="/private/tmp/xrt_stub"
if [[ ! -d "${STUB_DIR}/xrt" ]]; then
  echo "missing ${STUB_DIR}; create the stub headers or run the Codex preflight setup first"
  exit 1
fi
clang++ -std=c++17 -fsyntax-only -I "${STUB_DIR}" -I "${ROOT_DIR}/llamacpp_fpga" \
  "${ROOT_DIR}/scripts/host_driver.cpp" \
  "${ROOT_DIR}/llamacpp_fpga/smolvlm2_xrt_runner.cpp" \
  "${ROOT_DIR}/llamacpp_fpga/fpga_backend.cpp"

echo
echo "[5b/6] llama-server FPGA path static checks"
SERVER_CPP="${ROOT_DIR}/../llama.cpp/tools/server/server.cpp"
grep -q "smolvlm2_fpga_write_vision_patches_i8" "${SERVER_CPP}"
grep -q "smolvlm2_fpga_run_prefill" "${SERVER_CPP}"
grep -q "smolvlm2_fpga_run_decode_ttft" "${SERVER_CPP}"
grep -q "run_fpga_completion_task(task)" "${SERVER_CPP}"
if grep -q "run_fpga_completion_task(std::move(task))" "${SERVER_CPP}"; then
  echo "FPGA server path still moves task before fallback"
  exit 1
fi

echo
echo "[5c/6] bridge build/link references"
grep -q "smolvlm2_image_to_decoder_bridge_kernel" "${ROOT_DIR}/cfg/kv260_link_multi.cfg"
grep -q "hls_synth_bridge.tcl" "${ROOT_DIR}/scripts/build.sh"
grep -q "set_top smolvlm2_image_to_decoder_bridge_kernel" "${ROOT_DIR}/scripts/hls_synth_bridge.tcl"

echo
echo "[6/6] optional replay reminder"
echo "Run in the full export env after regenerating hardware_export:"
echo "  conda run -n <env-with-transformers> python scripts/replay_hardware_export.py --export-dir ${EXPORT_DIR} --layers all --check-lmhead"

echo
echo "Mac preflight PASS"
