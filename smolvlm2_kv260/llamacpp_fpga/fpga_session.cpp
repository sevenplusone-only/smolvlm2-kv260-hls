#include "fpga_session.h"

#include <stdexcept>

namespace smolvlm2 {

FpgaSession::FpgaSession(const FpgaSessionConfig &config)
    : config_(config),
      manifest_(load_hardware_manifest(config.hardware_export_dir)),
      backend_(std::make_shared<FpgaBackend>(config.xclbin_path, config.device_index)),
    runner_(std::make_unique<SmolVlm2XrtRunner>(backend_)) {
    if (config.max_seq == 0 || config.max_seq > kHwMaxSeq) {
        throw std::runtime_error(
            "invalid max_seq for FPGA session: model_max_seq=" +
            std::to_string(kModelMaxSeq) + " hw_max_seq=" + std::to_string(kHwMaxSeq)
        );
    }
    runner_->allocate_buffers(config.max_seq, manifest_);
    runner_->load_export_artifacts(config.hardware_export_dir);
}

void FpgaSession::write_vision_patch_activations(
    const int8_t *patches,
    size_t patch_count,
    size_t hidden_stride
) {
    runner_->write_vision_patch_activations(patches, patch_count, hidden_stride);
}

void FpgaSession::encode_image() {
    runner_->encode_image(manifest_);
}

int FpgaSession::build_prefill_activations(const std::vector<int> &token_ids) {
    return runner_->build_prefill_activations(token_ids);
}

int FpgaSession::build_prefill_activations(
    const std::vector<int> &token_ids,
    const int8_t *token_embeddings,
    size_t embedding_stride
) {
    return runner_->build_prefill_activations(token_ids, token_embeddings, embedding_stride);
}

void FpgaSession::run_prefill(int seq_len) {
    runner_->run_prefill_throughput(seq_len, manifest_.decoder_layers);
}

void FpgaSession::run_decode_ttft(int pos, int kv_len) {
    runner_->run_decode_ttft_token(pos, kv_len, manifest_.decoder_layers);
}

void FpgaSession::run_decode_window(int pos, int kv_len, int window_tokens) {
    runner_->run_decode_throughput_window(pos, kv_len, window_tokens, manifest_.decoder_layers);
}

void FpgaSession::write_decode_token_activation(int pos, const int8_t *embedding, size_t embedding_stride) {
    runner_->write_decode_token_activation(pos, embedding, embedding_stride);
}

std::vector<int32_t> FpgaSession::read_logits() {
    return runner_->read_logits();
}

} // namespace smolvlm2
