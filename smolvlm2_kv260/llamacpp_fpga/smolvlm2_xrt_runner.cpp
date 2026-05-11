#include "smolvlm2_xrt_runner.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace smolvlm2 {

static constexpr int kExecModeTtftDecode = 0;
static constexpr int kExecModeThroughputPrefill = 1;
static constexpr int kExecModeThroughputDecode = 2;
static constexpr int kPrefillTokenTile = 32;

static size_t file_size_or_throw(const std::string &path);

static void load_file_into_bo(const std::string &path, FpgaBuffer &buffer, size_t dst_offset = 0) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("failed to open artifact: " + path);
    }
    const size_t artifact_size = file_size_or_throw(path);
    if (dst_offset > buffer.bytes || artifact_size > buffer.bytes - dst_offset) {
        std::fclose(fp);
        throw std::runtime_error("artifact does not fit BO: " + path);
    }
    auto *dst = buffer.bo.map<uint8_t *>() + dst_offset;
    const size_t n = std::fread(dst, 1, artifact_size, fp);
    if (std::ferror(fp)) {
        std::fclose(fp);
        throw std::runtime_error("failed to read artifact: " + path);
    }
    std::fclose(fp);
    if (n != artifact_size) {
        throw std::runtime_error("short artifact read: " + path);
    }
}

static size_t file_size_or_throw(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("failed to open artifact: " + path);
    }
    std::fseek(fp, 0, SEEK_END);
    const long n = std::ftell(fp);
    std::fclose(fp);
    if (n < 0) {
        throw std::runtime_error("failed to stat artifact: " + path);
    }
    return (size_t)n;
}

SmolVlm2XrtRunner::SmolVlm2XrtRunner(std::shared_ptr<FpgaBackend> backend)
    : backend_(std::move(backend)) {}

void SmolVlm2XrtRunner::allocate_buffers(size_t max_seq) {
    if (max_seq > kHwMaxSeq) {
        throw std::runtime_error(
            "requested max_seq exceeds current hardware cap: model_max_seq=" +
            std::to_string(kModelMaxSeq) + " hw_max_seq=" + std::to_string(kHwMaxSeq)
        );
    }
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

void SmolVlm2XrtRunner::allocate_buffers(size_t max_seq, const HardwareManifest &manifest) {
    allocate_buffers(max_seq);
    auto require_fit = [](const char *name, size_t have, size_t need) {
        if (have < need) {
            throw std::runtime_error(std::string("BO too small for ") + name);
        }
    };
    require_fit("vision_wgt.bin", vision_wgt_.bytes, manifest.artifact_bytes.at("vision_wgt.bin"));
    require_fit("vision_meta.bin", vision_meta_.bytes, manifest.artifact_bytes.at("vision_meta.bin"));
    require_fit("connector_wgt.bin", connector_wgt_.bytes, manifest.artifact_bytes.at("connector_wgt.bin"));
    require_fit("connector_meta.bin", connector_meta_.bytes, manifest.artifact_bytes.at("connector_meta.bin"));
    require_fit("weights_half0.bin", decoder_wgt0_.bytes, manifest.artifact_bytes.at("weights_half0.bin"));
    require_fit("weights_half1.bin", decoder_wgt1_.bytes, manifest.artifact_bytes.at("weights_half1.bin"));
    require_fit("meta_half0.bin", decoder_meta0_.bytes, manifest.artifact_bytes.at("meta_half0.bin"));
    require_fit("meta_half1.bin", decoder_meta1_.bytes, manifest.artifact_bytes.at("meta_half1.bin"));
    require_fit("lmhead_wgt halves", lmhead_wgt_.bytes,
                manifest.artifact_bytes.at("lmhead_wgt_half0.bin") + manifest.artifact_bytes.at("lmhead_wgt_half1.bin"));
    require_fit("lmhead_meta halves", lmhead_meta_.bytes,
                manifest.artifact_bytes.at("lmhead_meta_half0.bin") + manifest.artifact_bytes.at("lmhead_meta_half1.bin"));
}

void SmolVlm2XrtRunner::load_export_artifacts(const std::string &export_dir) {
    auto path = [&](const char *name) {
        return export_dir + "/" + name;
    };

    load_file_into_bo(path("vision_wgt.bin"), vision_wgt_);
    backend_->sync_to_device(vision_wgt_);
    load_file_into_bo(path("vision_meta.bin"), vision_meta_);
    backend_->sync_to_device(vision_meta_);
    load_file_into_bo(path("connector_wgt.bin"), connector_wgt_);
    backend_->sync_to_device(connector_wgt_);
    load_file_into_bo(path("connector_meta.bin"), connector_meta_);
    backend_->sync_to_device(connector_meta_);
    load_file_into_bo(path("weights_half0.bin"), decoder_wgt0_);
    backend_->sync_to_device(decoder_wgt0_);
    load_file_into_bo(path("weights_half1.bin"), decoder_wgt1_);
    backend_->sync_to_device(decoder_wgt1_);
    load_file_into_bo(path("meta_half0.bin"), decoder_meta0_);
    backend_->sync_to_device(decoder_meta0_);
    load_file_into_bo(path("meta_half1.bin"), decoder_meta1_);
    backend_->sync_to_device(decoder_meta1_);
    load_file_into_bo(path("gamma_attn.bin"), gamma_attn_);
    backend_->sync_to_device(gamma_attn_);
    load_file_into_bo(path("gamma_ffn.bin"), gamma_ffn_);
    backend_->sync_to_device(gamma_ffn_);

    const size_t lm_w0 = file_size_or_throw(path("lmhead_wgt_half0.bin"));
    load_file_into_bo(path("lmhead_wgt_half0.bin"), lmhead_wgt_, 0);
    load_file_into_bo(path("lmhead_wgt_half1.bin"), lmhead_wgt_, lm_w0);
    backend_->sync_to_device(lmhead_wgt_);

    const size_t lm_m0 = file_size_or_throw(path("lmhead_meta_half0.bin"));
    load_file_into_bo(path("lmhead_meta_half0.bin"), lmhead_meta_, 0);
    load_file_into_bo(path("lmhead_meta_half1.bin"), lmhead_meta_, lm_m0);
    backend_->sync_to_device(lmhead_meta_);
}

void SmolVlm2XrtRunner::write_vision_patch_activations(
    const int8_t *patches,
    size_t patch_count,
    size_t hidden_stride
) {
    if (!patches) {
        throw std::runtime_error("vision patch activation pointer is null");
    }
    if (patch_count != (size_t)kVitTokens) {
        throw std::runtime_error("vision patch count must match 1024 ViT tokens");
    }
    if (hidden_stride < (size_t)kVitHidden) {
        throw std::runtime_error("vision patch hidden stride is smaller than 768");
    }

    auto *dst = vit_act_.bo.map<int8_t *>();
    for (size_t t = 0; t < patch_count; ++t) {
        std::memcpy(dst + t * kVitHidden, patches + t * hidden_stride, kVitHidden);
    }
    backend_->sync_to_device(vit_act_, kVitTokens * kVitHidden, 0);
}

void SmolVlm2XrtRunner::encode_image(const VisionOffsets &offsets) {
    auto &vit = backend_->vit_kernel();
    auto &conn = backend_->connector_kernel();

    // vit_act_ is supplied by the PS-side image preprocessor as quantized patch
    // activations.  ViT, connector and bridge stay device-side after this point.
    auto patch = vit(vit_act_.bo, vision_wgt_.bo, vision_meta_.bo, vit_hidden_.bo,
                     1, kVitTokens, kVitHidden, kVitHidden,
                     offsets.patch_wgt_axi, offsets.patch_meta_axi);
    patch.wait();

    auto qkv = vit(vit_hidden_.bo, vision_wgt_.bo, vision_meta_.bo, vit_tmp_.bo,
                   2, kVitTokens, kVitHidden, kVitHidden * 3,
                   offsets.qkv_wgt_axi, offsets.qkv_meta_axi);
    qkv.wait();

    auto attn = vit(vit_tmp_.bo, vision_wgt_.bo, vision_meta_.bo, vit_hidden_.bo,
                    3, kVitTokens, kVitHidden * 3, kVitHidden,
                    offsets.attn_wgt_axi, offsets.attn_meta_axi);
    attn.wait();

    auto conn_run = conn(vit_hidden_.bo, connector_wgt_.bo, connector_meta_.bo, image_tokens_.bo, 1,
                         offsets.connector_wgt_axi, offsets.connector_meta_axi);
    conn_run.wait();
}

void SmolVlm2XrtRunner::encode_image(const HardwareManifest &manifest) {
    encode_image(manifest.vision);
}

void SmolVlm2XrtRunner::prefill(int seq_len, const std::vector<WeightOffsets> &layer_offsets) {
    run_prefill_throughput(seq_len, layer_offsets);
}

void SmolVlm2XrtRunner::run_prefill_throughput(int seq_len, const std::vector<WeightOffsets> &layer_offsets) {
    auto &dec = backend_->decoder_kernel();
    if ((size_t)seq_len > max_seq_) {
        throw std::runtime_error("prefill seq_len exceeds allocated max_seq");
    }
    for (int layer = 0; layer < kNumLayers; ++layer) {
        const auto off = layer < (int)layer_offsets.size() ? layer_offsets[layer] : WeightOffsets{};
        for (int tile_offset = 0; tile_offset < seq_len; tile_offset += kPrefillTokenTile) {
            const int tile_size = std::min(kPrefillTokenTile, seq_len - tile_offset);
            bool use_ping = (layer % 2) == 0;
            auto run = dec(
                use_ping ? act_ping_.bo : act_pong_.bo,
                use_ping ? act_pong_.bo : act_ping_.bo,
                use_ping ? act_pong_.bo : act_ping_.bo,
                decoder_wgt0_.bo, decoder_wgt1_.bo,
                decoder_meta0_.bo, decoder_meta1_.bo,
                kv_k_.bo, kv_v_.bo,
                gamma_attn_.bo, gamma_ffn_.bo,
                lmhead_wgt_.bo, lmhead_meta_.bo, logits_.bo,
                seq_len, seq_len, 0,
                kExecModeThroughputPrefill, tile_offset, tile_size,
                0, layer, use_ping ? 1 : 0, off.wgt_axi, off.meta_axi,
                1, 15, 5
            );
            run.wait();
        }
    }
}

void SmolVlm2XrtRunner::prefill(int seq_len, const HardwareManifest &manifest) {
    prefill(seq_len, manifest.decoder_layers);
}

void SmolVlm2XrtRunner::decode_one(int pos, int kv_len, const std::vector<WeightOffsets> &layer_offsets) {
    run_decode_ttft_token(pos, kv_len, layer_offsets);
}

void SmolVlm2XrtRunner::run_decode_ttft_token(int pos, int kv_len, const std::vector<WeightOffsets> &layer_offsets) {
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
            1, kv_len + 1, pos,
            kExecModeTtftDecode, 0, 1,
            run_lmhead,
            layer, use_ping ? 1 : 0, off.wgt_axi, off.meta_axi,
            1, 15, 5
        );
        run.wait();
    }
}

void SmolVlm2XrtRunner::run_decode_throughput_window(
    int pos,
    int kv_len,
    int window_tokens,
    const std::vector<WeightOffsets> &layer_offsets
) {
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
            window_tokens, kv_len + window_tokens, pos,
            kExecModeThroughputDecode, 0, window_tokens,
            run_lmhead,
            layer, use_ping ? 1 : 0, off.wgt_axi, off.meta_axi,
            1, 15, 5
        );
        run.wait();
    }
}

void SmolVlm2XrtRunner::decode_one(int pos, int kv_len, const HardwareManifest &manifest) {
    decode_one(pos, kv_len, manifest.decoder_layers);
}

int SmolVlm2XrtRunner::build_prefill_activations(const std::vector<int> &token_ids) {
    return build_prefill_activations(token_ids, nullptr, 0);
}

int SmolVlm2XrtRunner::build_prefill_activations(
    const std::vector<int> &token_ids,
    const int8_t *token_embeddings,
    size_t embedding_stride
) {
    auto *image_ptr = image_tokens_.bo.map<int32_t *>();
    auto *act_ptr = act_ping_.bo.map<int8_t *>();
    std::memset(act_ptr, 0, act_ping_.bytes);

    int out_token = 0;
    size_t token_index = 0;
    bool saw_image_placeholder = false;
    int image_dst_token = 0;
    auto reserve_image_tokens = [&]() {
        if ((size_t)(out_token + kImageTokens) > max_seq_) {
            throw std::runtime_error("prefill image tokens exceed allocated max_seq");
        }
        image_dst_token = out_token;
        out_token += kImageTokens;
    };
    (void)image_ptr;
    if (token_embeddings && embedding_stride < (size_t)kLlmHidden) {
        throw std::runtime_error("token embedding stride is smaller than hidden size");
    }

    for (int token_id : token_ids) {
        if (token_id == kImageTokenId) {
            reserve_image_tokens();
            saw_image_placeholder = true;
            continue;
        }
        if ((size_t)out_token >= max_seq_) {
            throw std::runtime_error("prefill text tokens exceed allocated max_seq");
        }
        if (token_embeddings) {
            const int8_t *src = token_embeddings + token_index * embedding_stride;
            std::memcpy(act_ptr + (size_t)out_token * kLlmHidden, src, kLlmHidden);
        }
        out_token += 1;
        token_index += 1;
    }
    if (!saw_image_placeholder) {
        out_token = 0;
        reserve_image_tokens();
        if ((size_t)(out_token + token_ids.size()) > max_seq_) {
            throw std::runtime_error("prefill tokens exceed allocated max_seq after prepending image tokens");
        }
        if (token_embeddings) {
            for (size_t i = 0; i < token_ids.size(); ++i) {
                const int8_t *src = token_embeddings + i * embedding_stride;
                std::memcpy(act_ptr + (size_t)(out_token + (int)i) * kLlmHidden, src, kLlmHidden);
            }
        }
        out_token += (int)token_ids.size();
    }

    if ((size_t)out_token > max_seq_) {
        throw std::runtime_error(
            "prefill token count exceeds hardware cap: model_max_seq=" +
            std::to_string(kModelMaxSeq) + " hw_max_seq=" + std::to_string((int)max_seq_)
        );
    }
    backend_->sync_to_device(act_ping_);
    run_image_bridge(image_dst_token, kImageTokens);
    return out_token;
}

void SmolVlm2XrtRunner::write_decode_token_activation(int pos, const int8_t *embedding, size_t embedding_stride) {
    if (!embedding) {
        throw std::runtime_error("decode token embedding is null");
    }
    if (pos < 0 || (size_t)pos >= max_seq_) {
        throw std::runtime_error("decode token position exceeds allocated max_seq");
    }
    if (embedding_stride < (size_t)kLlmHidden) {
        throw std::runtime_error("decode token embedding stride is smaller than hidden size");
    }
    auto *act_ptr = act_ping_.bo.map<int8_t *>();
    const size_t offset = (size_t)pos * kLlmHidden;
    std::memcpy(act_ptr + offset, embedding, kLlmHidden);
    backend_->sync_to_device(act_ping_, kLlmHidden, offset);
}

void SmolVlm2XrtRunner::run_image_bridge(int dst_token_offset, int num_image_tokens) {
    auto &bridge = backend_->bridge_kernel();
    auto run = bridge(image_tokens_.bo, act_ping_.bo, dst_token_offset, num_image_tokens, kLlmHidden);
    run.wait();
}

std::vector<int32_t> SmolVlm2XrtRunner::read_logits() {
    backend_->sync_from_device(logits_);
    auto *ptr = logits_.bo.map<int32_t *>();
    return std::vector<int32_t>(ptr, ptr + kVocabSize);
}

} // namespace smolvlm2
