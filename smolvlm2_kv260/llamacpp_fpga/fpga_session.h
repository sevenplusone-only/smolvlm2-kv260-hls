#pragma once

#include "fpga_backend.h"
#include "hardware_manifest.h"
#include "smolvlm2_xrt_runner.h"

#include <memory>
#include <string>
#include <vector>

namespace smolvlm2 {

struct FpgaSessionConfig {
    std::string xclbin_path;
    std::string hardware_export_dir;
    size_t max_seq = kHwMaxSeq;
    unsigned device_index = 0;
};

class FpgaSession {
public:
    explicit FpgaSession(const FpgaSessionConfig &config);

    const HardwareManifest &manifest() const { return manifest_; }
    SmolVlm2XrtRunner &runner() { return *runner_; }
    const FpgaSessionConfig &config() const { return config_; }

    void write_vision_patch_activations(const int8_t *patches, size_t patch_count, size_t hidden_stride);
    void encode_image();
    int build_prefill_activations(const std::vector<int> &token_ids);
    int build_prefill_activations(const std::vector<int> &token_ids, const int8_t *token_embeddings, size_t embedding_stride);
    void run_prefill(int seq_len);
    void run_decode_ttft(int pos, int kv_len);
    void run_decode_window(int pos, int kv_len, int window_tokens);
    void write_decode_token_activation(int pos, const int8_t *embedding, size_t embedding_stride);
    std::vector<int32_t> read_logits();

private:
    FpgaSessionConfig config_;
    HardwareManifest manifest_;
    std::shared_ptr<FpgaBackend> backend_;
    std::unique_ptr<SmolVlm2XrtRunner> runner_;
};

} // namespace smolvlm2
