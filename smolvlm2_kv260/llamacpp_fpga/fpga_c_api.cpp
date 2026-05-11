#include "fpga_c_api.h"

#include "fpga_session.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct smolvlm2_fpga_session {
    std::unique_ptr<smolvlm2::FpgaSession> impl;
};

static void write_error(const std::string &msg, char *buf, size_t buf_size) {
    if (!buf || buf_size == 0) {
        return;
    }
    std::snprintf(buf, buf_size, "%s", msg.c_str());
}

extern "C" {

struct smolvlm2_fpga_session * smolvlm2_fpga_session_create(
    const struct smolvlm2_fpga_session_config * config,
    char * error_buf,
    size_t error_buf_size
) {
    if (!config || !config->xclbin_path || !config->hardware_export_dir) {
        write_error("invalid fpga session config", error_buf, error_buf_size);
        return nullptr;
    }

    try {
        smolvlm2::FpgaSessionConfig cpp_cfg;
        cpp_cfg.xclbin_path = config->xclbin_path;
        cpp_cfg.hardware_export_dir = config->hardware_export_dir;
        cpp_cfg.max_seq = config->max_seq ? config->max_seq : smolvlm2::kHwMaxSeq;
        cpp_cfg.device_index = config->device_index;

        auto *session = new smolvlm2_fpga_session;
        session->impl = std::make_unique<smolvlm2::FpgaSession>(cpp_cfg);
        return session;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return nullptr;
    }
}

void smolvlm2_fpga_session_destroy(struct smolvlm2_fpga_session * session) {
    delete session;
}

int smolvlm2_fpga_write_vision_patches_i8(
    struct smolvlm2_fpga_session * session,
    const int8_t * patches,
    size_t patch_count,
    size_t hidden_stride,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        session->impl->write_vision_patch_activations(patches, patch_count, hidden_stride);
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

int smolvlm2_fpga_encode_image(
    struct smolvlm2_fpga_session * session,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        session->impl->encode_image();
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

int smolvlm2_fpga_build_prefill_activations(
    struct smolvlm2_fpga_session * session,
    const int32_t * token_ids,
    size_t n_tokens,
    int * out_seq_len,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        std::vector<int> tokens;
        tokens.reserve(n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.push_back((int)token_ids[i]);
        }
        const int seq_len = session->impl->build_prefill_activations(tokens);
        if (out_seq_len) {
            *out_seq_len = seq_len;
        }
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

int smolvlm2_fpga_build_prefill_activations_i8(
    struct smolvlm2_fpga_session * session,
    const int32_t * token_ids,
    size_t n_tokens,
    const int8_t * token_embeddings,
    size_t embedding_stride,
    int * out_seq_len,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        std::vector<int> tokens;
        tokens.reserve(n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.push_back((int)token_ids[i]);
        }
        const int seq_len = session->impl->build_prefill_activations(tokens, token_embeddings, embedding_stride);
        if (out_seq_len) {
            *out_seq_len = seq_len;
        }
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

int smolvlm2_fpga_run_prefill(
    struct smolvlm2_fpga_session * session,
    int seq_len,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        session->impl->run_prefill(seq_len);
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

int smolvlm2_fpga_run_decode_ttft(
    struct smolvlm2_fpga_session * session,
    int pos,
    int kv_len,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        session->impl->run_decode_ttft(pos, kv_len);
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

int smolvlm2_fpga_run_decode_window(
    struct smolvlm2_fpga_session * session,
    int pos,
    int kv_len,
    int window_tokens,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        session->impl->run_decode_window(pos, kv_len, window_tokens);
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

int smolvlm2_fpga_write_decode_token_activation(
    struct smolvlm2_fpga_session * session,
    int pos,
    const int8_t * embedding,
    size_t embedding_stride,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        session->impl->write_decode_token_activation(pos, embedding, embedding_stride);
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

int smolvlm2_fpga_read_logits(
    struct smolvlm2_fpga_session * session,
    int32_t * out_logits,
    size_t logits_count,
    char * error_buf,
    size_t error_buf_size
) {
    try {
        const auto logits = session->impl->read_logits();
        if (logits_count < logits.size()) {
            write_error("output logits buffer too small", error_buf, error_buf_size);
            return -1;
        }
        std::copy(logits.begin(), logits.end(), out_logits);
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), error_buf, error_buf_size);
        return -1;
    }
}

} // extern "C"
