#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct smolvlm2_fpga_session;

struct smolvlm2_fpga_session_config {
    const char * xclbin_path;
    const char * hardware_export_dir;
    size_t max_seq;
    unsigned device_index;
};

struct smolvlm2_fpga_session * smolvlm2_fpga_session_create(
    const struct smolvlm2_fpga_session_config * config,
    char * error_buf,
    size_t error_buf_size
);

void smolvlm2_fpga_session_destroy(struct smolvlm2_fpga_session * session);

int smolvlm2_fpga_write_vision_patches_i8(
    struct smolvlm2_fpga_session * session,
    const int8_t * patches,
    size_t patch_count,
    size_t hidden_stride,
    char * error_buf,
    size_t error_buf_size
);

int smolvlm2_fpga_encode_image(
    struct smolvlm2_fpga_session * session,
    char * error_buf,
    size_t error_buf_size
);

int smolvlm2_fpga_build_prefill_activations(
    struct smolvlm2_fpga_session * session,
    const int32_t * token_ids,
    size_t n_tokens,
    int * out_seq_len,
    char * error_buf,
    size_t error_buf_size
);

int smolvlm2_fpga_build_prefill_activations_i8(
    struct smolvlm2_fpga_session * session,
    const int32_t * token_ids,
    size_t n_tokens,
    const int8_t * token_embeddings,
    size_t embedding_stride,
    int * out_seq_len,
    char * error_buf,
    size_t error_buf_size
);

int smolvlm2_fpga_run_prefill(
    struct smolvlm2_fpga_session * session,
    int seq_len,
    char * error_buf,
    size_t error_buf_size
);

int smolvlm2_fpga_run_decode_ttft(
    struct smolvlm2_fpga_session * session,
    int pos,
    int kv_len,
    char * error_buf,
    size_t error_buf_size
);

int smolvlm2_fpga_run_decode_window(
    struct smolvlm2_fpga_session * session,
    int pos,
    int kv_len,
    int window_tokens,
    char * error_buf,
    size_t error_buf_size
);

int smolvlm2_fpga_write_decode_token_activation(
    struct smolvlm2_fpga_session * session,
    int pos,
    const int8_t * embedding,
    size_t embedding_stride,
    char * error_buf,
    size_t error_buf_size
);

int smolvlm2_fpga_read_logits(
    struct smolvlm2_fpga_session * session,
    int32_t * out_logits,
    size_t logits_count,
    char * error_buf,
    size_t error_buf_size
);

#ifdef __cplusplus
}
#endif
