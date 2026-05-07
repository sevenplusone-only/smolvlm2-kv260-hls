#include "smolvlm2_xrt_runner.h"

#include <cstring>

namespace smolvlm2 {

SmolVlm2XrtRunner::SmolVlm2XrtRunner(std::shared_ptr<FpgaBackend> backend)
    : backend_(std::move(backend)) {}

void SmolVlm2XrtRunner::allocate_buffers(size_t max_seq) {
    max_seq_ = max_seq;

    auto &vit = backend_->vit_kernel();
    auto &conn = backend_->connector_kernel();
    auto &dec = backend_->decoder_kernel();

    vit_act_ = backend_->alloc(kVitTokens * kVitHidden, vit, 0);
    vit_hidden_ = backend_->alloc(kVitTokens * kVitHidden * sizeof(int32_t), vit, 3);
    vit_tmp_ = backend_->alloc(kVitTokens * 3072 * sizeof(int32_t), vit, 3);
    vision_wgt_ = backend_->alloc(128 * 1024 * 1024, vit, 1);
    vision_meta_ = backend_->alloc(16 * 1024 * 1024, vit, 2);
    connector_wgt_ = backend_->alloc((size_t)kConnectorIn * kLlmHidden / 2, conn, 1);
    connector_meta_ = backend_->alloc((size_t)kConnectorIn * kLlmHidden / 32 * 2, conn, 2);
    image_tokens_ = backend_->alloc(kImageTokens * kLlmHidden * sizeof(int32_t), conn, 3);

    act_ping_ = backend_->alloc(max_seq * kLlmHidden, dec, 0);
    act_pong_ = backend_->alloc(max_seq * kLlmHidden, dec, 1);
    act_out_ = backend_->alloc(max_seq * kLlmHidden, dec, 2);
    const size_t layer_wgt_bytes =
        (size_t)kLlmHidden * (kLlmHidden + kNumKvHeads * 64 * 2) / 2 +
        (size_t)kLlmHidden * kLlmHidden / 2 +
        (size_t)kLlmHidden * 2560 * 2 / 2 +
        (size_t)2560 * kLlmHidden / 2;
    const size_t decoder_wgt_bytes = layer_wgt_bytes * kNumLayers;
    const size_t decoder_meta_bytes = decoder_wgt_bytes / 32 * 2;
    decoder_wgt0_ = backend_->alloc(decoder_wgt_bytes / 2, dec, 3);
    decoder_wgt1_ = backend_->alloc(decoder_wgt_bytes / 2, dec, 4);
    decoder_meta0_ = backend_->alloc(decoder_meta_bytes / 2, dec, 5);
    decoder_meta1_ = backend_->alloc(decoder_meta_bytes / 2, dec, 6);
    kv_k_ = backend_->alloc(kNumLayers * kNumKvHeads * 2048 * 64 / 2, dec, 7);
    kv_v_ = backend_->alloc(kNumLayers * kNumKvHeads * 2048 * 64 / 2, dec, 8);
    gamma_attn_ = backend_->alloc(kLlmHidden, dec, 9);
    gamma_ffn_ = backend_->alloc(kLlmHidden, dec, 10);
    lmhead_wgt_ = backend_->alloc((size_t)kLlmHidden * kVocabSize / 2, dec, 11);
    lmhead_meta_ = backend_->alloc((size_t)kLlmHidden * kVocabSize / 32 * 2, dec, 12);
    logits_ = backend_->alloc(kVocabSize * sizeof(int32_t), dec, 13);
}

void SmolVlm2XrtRunner::encode_image() {
    auto &vit = backend_->vit_kernel();
    auto &conn = backend_->connector_kernel();

    // The official llama.cpp front-end should fill vit_act_ with preprocessed
    // patch activations.  Weight BOs are supplied by the quantized manifest
    // loader; until then this runner defines the XRT schedule and buffer shape.
    auto patch = vit(vit_act_.bo, vision_wgt_.bo, vision_meta_.bo, vit_hidden_.bo,
                     1, kVitTokens, kVitHidden, kVitHidden);
    patch.wait();

    auto qkv = vit(vit_hidden_.bo, vision_wgt_.bo, vision_meta_.bo, vit_tmp_.bo,
                   2, kVitTokens, kVitHidden, kVitHidden * 3);
    qkv.wait();

    auto attn = vit(vit_tmp_.bo, vision_wgt_.bo, vision_meta_.bo, vit_hidden_.bo,
                    3, kVitTokens, kVitHidden * 3, kVitHidden);
    attn.wait();

    auto conn_run = conn(vit_hidden_.bo, connector_wgt_.bo, connector_meta_.bo, image_tokens_.bo, 1);
    conn_run.wait();
}

void SmolVlm2XrtRunner::prefill(int seq_len, const std::vector<WeightOffsets> &layer_offsets) {
    auto &dec = backend_->decoder_kernel();
    for (int layer = 0; layer < kNumLayers; ++layer) {
        bool use_ping = (layer % 2) == 0;
        const auto off = layer < (int)layer_offsets.size() ? layer_offsets[layer] : WeightOffsets{};
        auto run = dec(
            use_ping ? act_ping_.bo : act_pong_.bo,
            use_ping ? act_pong_.bo : act_ping_.bo,
            use_ping ? act_pong_.bo : act_ping_.bo,
            decoder_wgt0_.bo, decoder_wgt1_.bo,
            decoder_meta0_.bo, decoder_meta1_.bo,
            kv_k_.bo, kv_v_.bo,
            gamma_attn_.bo, gamma_ffn_.bo,
            lmhead_wgt_.bo, lmhead_meta_.bo, logits_.bo,
            seq_len, seq_len, 0, 0,
            layer, use_ping ? 1 : 0, off.wgt_axi, off.meta_axi,
            1, 15, 5
        );
        run.wait();
    }
}

void SmolVlm2XrtRunner::decode_one(int pos, int kv_len, const std::vector<WeightOffsets> &layer_offsets) {
    auto &dec = backend_->decoder_kernel();
    for (int layer = 0; layer < kNumLayers; ++layer) {
        bool use_ping = (layer % 2) == 0;
        int run_lmhead = layer == kNumLayers - 1 ? 1 : 0;
        const auto off = layer < (int)layer_offsets.size() ? layer_offsets[layer] : WeightOffsets{};
        auto run = dec(
            use_ping ? act_ping_.bo : act_pong_.bo,
            use_ping ? act_pong_.bo : act_ping_.bo,
            use_ping ? act_pong_.bo : act_ping_.bo,
            decoder_wgt0_.bo, decoder_wgt1_.bo,
            decoder_meta0_.bo, decoder_meta1_.bo,
            kv_k_.bo, kv_v_.bo,
            gamma_attn_.bo, gamma_ffn_.bo,
            lmhead_wgt_.bo, lmhead_meta_.bo, logits_.bo,
            1, kv_len, pos, run_lmhead,
            layer, use_ping ? 1 : 0, off.wgt_axi, off.meta_axi,
            1, 15, 5
        );
        run.wait();
    }
}

std::vector<int32_t> SmolVlm2XrtRunner::read_logits() {
    backend_->sync_from_device(logits_);
    auto *ptr = logits_.bo.map<int32_t *>();
    return std::vector<int32_t>(ptr, ptr + kVocabSize);
}

} // namespace smolvlm2
