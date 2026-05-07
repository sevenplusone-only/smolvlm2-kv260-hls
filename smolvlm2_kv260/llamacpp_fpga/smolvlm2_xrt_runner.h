#pragma once

#include "fpga_backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace smolvlm2 {

static constexpr int kImageTokenId = 49190;
static constexpr int kVitTokens = 1024;
static constexpr int kVitHidden = 768;
static constexpr int kImageTokens = 64;
static constexpr int kConnectorIn = 12288;
static constexpr int kLlmHidden = 960;
static constexpr int kVocabSize = 49280;
static constexpr int kNumLayers = 32;
static constexpr int kNumKvHeads = 5;

struct WeightOffsets {
    int wgt_axi = 0;
    int meta_axi = 0;
};

class SmolVlm2XrtRunner {
public:
    explicit SmolVlm2XrtRunner(std::shared_ptr<FpgaBackend> backend);

    void allocate_buffers(size_t max_seq);
    void encode_image();
    void prefill(int seq_len, const std::vector<WeightOffsets> &layer_offsets);
    void decode_one(int pos, int kv_len, const std::vector<WeightOffsets> &layer_offsets);
    std::vector<int32_t> read_logits();

    FpgaBuffer &text_or_prefill_activations() { return act_ping_; }
    FpgaBuffer &image_embeddings() { return image_tokens_; }

private:
    std::shared_ptr<FpgaBackend> backend_;
    size_t max_seq_ = 0;

    FpgaBuffer vit_act_;
    FpgaBuffer vit_hidden_;
    FpgaBuffer vit_tmp_;
    FpgaBuffer vision_wgt_;
    FpgaBuffer vision_meta_;
    FpgaBuffer connector_wgt_;
    FpgaBuffer connector_meta_;
    FpgaBuffer image_tokens_;
    FpgaBuffer act_ping_;
    FpgaBuffer act_pong_;
    FpgaBuffer act_out_;
    FpgaBuffer decoder_wgt0_;
    FpgaBuffer decoder_wgt1_;
    FpgaBuffer decoder_meta0_;
    FpgaBuffer decoder_meta1_;
    FpgaBuffer kv_k_;
    FpgaBuffer kv_v_;
    FpgaBuffer gamma_attn_;
    FpgaBuffer gamma_ffn_;
    FpgaBuffer lmhead_wgt_;
    FpgaBuffer lmhead_meta_;
    FpgaBuffer logits_;
};

} // namespace smolvlm2
