#include "weight-pack.h"
#include "ggml-impl.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <array>

// GGUF block_q8_0 on-disk layout (must match ggml-common.h):
//   struct block_q8_0 {
//     ggml_half d;       // fp16 scale (2 bytes)
//     int8_t    qs[32];  // 32 int8 quants
//   }; // total 34 bytes per block
struct gguf_block_q8_0 {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(gguf_block_q8_0) == 34, "block_q8_0 size must be 34 bytes");

namespace ggml_xrt {

const sidecar_tensor * sidecar_manifest::find(const char * name) const {
    if (name == nullptr) {
        return nullptr;
    }
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
}

const sidecar_tensor * sidecar_manifest::find_alias(const char * name) const {
    const sidecar_tensor * exact = find(name);
    if (exact != nullptr || name == nullptr) {
        return exact;
    }

    const std::string n(name);
    std::array<std::string, 6> aliases{};
    size_t n_aliases = 0;

    auto push_alias = [&](std::string alias) {
        if (!alias.empty()) {
            aliases[n_aliases++] = std::move(alias);
        }
    };

    if (n == "output.weight") {
        push_alias("output.weight");
        push_alias("output");
    } else if (n == "output") {
        push_alias("output.weight");
    } else if (n == "mm.model.fc.weight") {
        push_alias("mm.0.weight");
        push_alias("mm.1.weight");
    } else if (n == "v.patch_embd.weight") {
        push_alias("vision.patch_embed");
    } else if (n.rfind("v.blk.", 0) == 0) {
        std::string alias = n;
        const std::string attn_out = ".attn_out.weight";
        const std::string attn_output = ".attn_output.weight";
        if (alias.size() > attn_out.size() &&
            alias.compare(alias.size() - attn_out.size(), attn_out.size(), attn_out) == 0) {
            alias.replace(alias.size() - attn_out.size(), attn_out.size(), attn_output);
            push_alias(alias);
        } else if (alias.size() > attn_output.size() &&
                   alias.compare(alias.size() - attn_output.size(), attn_output.size(), attn_output) == 0) {
            alias.replace(alias.size() - attn_output.size(), attn_output.size(), attn_out);
            push_alias(alias);
        }
    }

    for (size_t i = 0; i < n_aliases; ++i) {
        auto it = tensors.find(aliases[i]);
        if (it != tensors.end()) {
            return &it->second;
        }
    }

    return nullptr;
}

void unpack_gguf_q8_to_split(
    const void * src,
    int8_t *     int8_out,
    uint16_t *   fp16_scale_out,
    size_t       N,
    size_t       K)
{
    GGML_ASSERT((K % 32) == 0);
    const size_t K_G = K / 32;
    const gguf_block_q8_0 * blocks = (const gguf_block_q8_0 *)src;
    for (size_t n = 0; n < N; ++n) {
        for (size_t kg = 0; kg < K_G; ++kg) {
            const gguf_block_q8_0 & b = blocks[n*K_G + kg];
            fp16_scale_out[n*K_G + kg] = b.d;
            std::memcpy(&int8_out[n*K + kg*32], b.qs, 32);
        }
    }
}

void unpack_q8_0_blob_to_split(
    const uint8_t * src,
    int8_t *        int8_out,
    uint16_t *      fp16_scale_out,
    size_t          N,
    size_t          K)
{
    unpack_gguf_q8_to_split(src, int8_out, fp16_scale_out, N, K);
}

sidecar_manifest load_sidecar_manifest(const std::string & manifest_path) {
    sidecar_manifest result;

    if (manifest_path.empty()) {
        return result;
    }

    std::ifstream manifest_file(manifest_path);
    if (!manifest_file) {
        return result;
    }

    nlohmann::ordered_json json = nlohmann::ordered_json::parse(manifest_file, nullptr, false);
    if (json.is_discarded() || !json.contains("native_backend")) {
        return result;
    }

    result.source_model = json.value("source_model", "");

    const auto & native = json["native_backend"];
    result.weights_file = native.value("weights_file", "");
    if (result.weights_file.empty()) {
        return result;
    }

    std::string blob_path = result.weights_file;
    const auto slash = manifest_path.find_last_of("/\\");
    if (slash != std::string::npos && blob_path.find('/') == std::string::npos && blob_path.find('\\') == std::string::npos) {
        blob_path = manifest_path.substr(0, slash + 1) + blob_path;
    }

    std::ifstream blob_file(blob_path, std::ios::binary);
    if (!blob_file) {
        return sidecar_manifest{};
    }
    blob_file.seekg(0, std::ios::end);
    const auto blob_size = (size_t) blob_file.tellg();
    blob_file.seekg(0, std::ios::beg);
    result.weights_blob.resize(blob_size);
    if (blob_size > 0) {
        blob_file.read(reinterpret_cast<char *>(result.weights_blob.data()), (std::streamsize) blob_size);
    }

    if (!native.contains("tensors") || !native["tensors"].is_array()) {
        return sidecar_manifest{};
    }

    for (const auto & entry : native["tensors"]) {
        sidecar_tensor tensor;
        tensor.name         = entry.value("name", "");
        tensor.domain       = entry.value("domain", "");
        tensor.offset_bytes = entry.value("offset_bytes", (size_t) 0);
        tensor.nbytes       = entry.value("nbytes", (size_t) 0);
        tensor.K            = entry.value("K", (size_t) 0);
        tensor.N            = entry.value("N", (size_t) 0);
        tensor.quant        = entry.value("quant", "");

        if (tensor.name.empty() || tensor.nbytes == 0) {
            continue;
        }
        if (tensor.offset_bytes + tensor.nbytes > result.weights_blob.size()) {
            GGML_LOG_WARN("ggml-xrt: sidecar tensor %s exceeds blob bounds, skipping\n", tensor.name.c_str());
            continue;
        }
        result.tensors.emplace(tensor.name, std::move(tensor));
    }

    return result;
}

} // namespace ggml_xrt
