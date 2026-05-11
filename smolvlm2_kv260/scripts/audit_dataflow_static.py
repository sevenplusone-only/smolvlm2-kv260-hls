#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "src" / "decoder_kernel.cpp",
    ROOT / "src" / "attention.h",
    ROOT / "src" / "vit_kernel.cpp",
    ROOT / "src" / "w4a8_gemm.h",
    ROOT / "scripts" / "host_driver.cpp",
    ROOT / "llamacpp_fpga" / "smolvlm2_xrt_runner.cpp",
    ROOT / "llamacpp_fpga" / "fpga_session.cpp",
    ROOT / "src" / "bridge_kernel.cpp",
    ROOT.parents[0] / "llama.cpp" / "tools" / "server" / "server.cpp",
]


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def main() -> int:
    warnings: list[str] = []
    print("=== Static Dataflow Audit ===")
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            rel = path.relative_to(ROOT.parents[0])
        bundles = re.findall(r"bundle=([A-Za-z0-9_]+)", text)
        print(f"{rel}:")
        print(f"  m_axi pragmas: {count(r'HLS INTERFACE m_axi', text)}")
        if bundles:
            uniq = sorted(set(bundles))
            print(f"  bundles: {', '.join(uniq)}")
        print(f"  BO sync to device:   {count(r'XCL_BO_SYNC_BO_TO_DEVICE|sync_to_device', text)}")
        print(f"  BO sync from device: {count(r'XCL_BO_SYNC_BO_FROM_DEVICE|sync_from_device', text)}")

    attention = (ROOT / "src" / "attention.h").read_text(encoding="utf-8")
    decoder = (ROOT / "src" / "decoder_kernel.cpp").read_text(encoding="utf-8")
    if "axi_write_128_half(k_cache" not in attention or "axi_write_128_half(v_cache" not in attention:
        warnings.append("attention.h does not appear to write packed K/V cache to DDR")
    if "kv_cache_load_head" not in attention:
        warnings.append("attention.h does not appear to reload historical KV cache")
    if "layer_id * H_KV * MAX_SEQ" not in attention:
        warnings.append("attention.h KV cache address does not appear to include layer_id")
    if "MAX_SEQ * (D_HEAD / GRP)" not in attention:
        warnings.append("attention.h kv_buf depth may not cover all KV groups")
    if "#pragma HLS DATAFLOW" not in decoder:
        warnings.append("decoder_kernel.cpp appears to have lost DATAFLOW")
    if "hls::stream<INT32>" not in decoder or "hls::stream<INT8>" not in decoder:
        warnings.append("decoder_kernel.cpp appears to have lost stream/FIFO structure")

    host = (ROOT / "scripts" / "host_driver.cpp").read_text(encoding="utf-8")
    runner = (ROOT / "llamacpp_fpga" / "smolvlm2_xrt_runner.cpp").read_text(encoding="utf-8")
    session = (ROOT / "llamacpp_fpga" / "fpga_session.cpp").read_text(encoding="utf-8")
    server = (ROOT.parents[0] / "llama.cpp" / "tools" / "server" / "server.cpp").read_text(encoding="utf-8")
    if "image_tokens_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE)" in host:
        warnings.append("host_driver.cpp still syncs image_tokens back to CPU in the default path")
    if "backend_->sync_from_device(image_tokens_)" in runner:
        warnings.append("smolvlm2_xrt_runner.cpp still syncs image_tokens back to CPU in the default path")
    if "smolvlm2_image_to_decoder_bridge_kernel" not in (ROOT / "src" / "bridge_kernel.cpp").read_text(encoding="utf-8"):
        warnings.append("bridge kernel source is missing")
    if "bridge_kernel_" not in host:
        warnings.append("host_driver.cpp does not appear to launch bridge kernel")
    if "bridge_kernel()" not in runner:
        warnings.append("smolvlm2_xrt_runner.cpp does not appear to launch bridge kernel")
    if "run_decode_ttft_token" in host and "kv_len + 1" not in host:
        warnings.append("host_driver.cpp decode may pass kv_len without current token")
    if "kv_len + 1" not in runner:
        warnings.append("smolvlm2_xrt_runner.cpp decode may pass kv_len without current token")
    if "run_prefill_throughput" not in host or "run_decode_ttft_token" not in host or "run_decode_throughput_window" not in host:
        warnings.append("host_driver.cpp does not expose all v2 path skeletons")
    if "run_prefill_throughput" not in runner or "run_decode_ttft_token" not in runner or "run_decode_throughput_window" not in runner:
        warnings.append("smolvlm2_xrt_runner.cpp does not expose all v2 path skeletons")
    if "FpgaSession::FpgaSession" not in session or "load_hardware_manifest" not in session:
        warnings.append("fpga_session.cpp does not appear to glue manifest/backend/runner together")
    if "model_max_seq" not in host.lower() or "hw_max_seq" not in host.lower():
        warnings.append("host_driver.cpp does not appear to print/guard model_max_seq vs hw_max_seq")
    if "run_fpga_completion_task(std::move(task))" in server:
        warnings.append("llama-server FPGA path moves task before CPU fallback")
    if "run_fpga_completion_task(const server_task & task)" not in server:
        warnings.append("llama-server FPGA completion path is not const-task based")
    if "smolvlm2_fpga_write_vision_patches_i8" not in server:
        warnings.append("llama-server does not write preprocessed vision patches into FPGA vit_act")
    if "smolvlm2_fpga_run_prefill" not in server or "smolvlm2_fpga_run_decode_ttft" not in server:
        warnings.append("llama-server does not appear to call FPGA prefill/decode")
    fpga_block_match = re.search(r"bool run_fpga_completion_task\(const server_task & task\)(.*?)\n    }\n#endif", server, re.S)
    if not fpga_block_match:
        warnings.append("could not locate llama-server FPGA completion block")
    elif "llama_decode" in fpga_block_match.group(1) or "process_chunk" in fpga_block_match.group(1):
        warnings.append("llama-server FPGA completion path still calls CPU/GGML decode helpers")

    print()
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f" - {warning}")
        return 1
    print("Static dataflow audit PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
