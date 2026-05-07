#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml_xrt {

struct sidecar_tensor {
    std::string name;
    std::string domain;
    size_t      offset_bytes = 0;
    size_t      nbytes = 0;
    size_t      K = 0;
    size_t      N = 0;
    std::string quant;
};

struct sidecar_manifest {
    std::string source_model;
    std::string weights_file;
    std::unordered_map<std::string, sidecar_tensor> tensors;
    std::vector<uint8_t> weights_blob;

    bool empty() const { return tensors.empty() || weights_blob.empty(); }
    const sidecar_tensor * find(const char * name) const;
    const sidecar_tensor * find_alias(const char * name) const;
};

// Unpack GGUF block_q8_0 tensor (layout: [N][K/32] blocks of {fp16 scale, 32×int8 quants})
// into two dense arrays:
//   int8_out[N * K]        — row-major [N][K] int8 payload
//   fp16_scale_out[N*K/32] — row-major [N][K/32] fp16 scales
// src is the raw GGUF tensor data pointer. K must be a multiple of 32.
void unpack_gguf_q8_to_split(
    const void * src,
    int8_t *     int8_out,
    uint16_t *   fp16_scale_out,
    size_t       N,
    size_t       K);

sidecar_manifest load_sidecar_manifest(const std::string & manifest_path);

void unpack_q8_0_blob_to_split(
    const uint8_t * src,
    int8_t *        int8_out,
    uint16_t *      fp16_scale_out,
    size_t          N,
    size_t          K);

} // namespace ggml_xrt
