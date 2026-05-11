#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace smolvlm2 {

static constexpr int kAxi256Bytes = 32;

struct WeightOffsets {
    int wgt_axi = 0;
    int meta_axi = 0;
    size_t wgt_bytes = 0;
    size_t meta_bytes = 0;
};

struct VisionOffsets {
    int patch_wgt_axi = 0;
    int patch_meta_axi = 0;
    int qkv_wgt_axi = 0;
    int qkv_meta_axi = 0;
    int attn_wgt_axi = 0;
    int attn_meta_axi = 0;
    int connector_wgt_axi = 0;
    int connector_meta_axi = 0;
};

struct VisionLayerManifest {
    std::string name;
    int wgt_axi = 0;
    int meta_axi = 0;
    int M = 0;
    int K = 0;
    int N = 0;
};

struct ConnectorManifest {
    std::string name;
    int wgt_axi = 0;
    int meta_axi = 0;
    int M = 0;
    int K = 0;
    int N = 0;
};

struct HardwareManifest {
    std::string export_dir;
    std::string kv260_status;
    std::vector<WeightOffsets> decoder_layers;
    std::vector<VisionLayerManifest> vision_layers;
    ConnectorManifest connector;
    VisionOffsets vision;
    std::map<std::string, size_t> artifact_bytes;
};

inline std::string manifest_slurp(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

inline size_t manifest_file_size(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("failed to open artifact: " + path);
    }
    return (size_t)ifs.tellg();
}

inline std::string manifest_extract_string(
    const std::string &blob,
    const std::string &name,
    size_t from_pos = 0
) {
    const std::string key = "\"" + name + "\"";
    const size_t key_pos = blob.find(key, from_pos);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing key in manifest: " + name);
    }
    const size_t colon = blob.find(':', key_pos);
    const size_t begin_quote = blob.find('"', colon + 1);
    const size_t end_quote = blob.find('"', begin_quote + 1);
    if (begin_quote == std::string::npos || end_quote == std::string::npos) {
        throw std::runtime_error("malformed string key in manifest: " + name);
    }
    return blob.substr(begin_quote + 1, end_quote - begin_quote - 1);
}

inline int manifest_extract_int(
    const std::string &blob,
    const std::string &name,
    size_t from_pos = 0
) {
    const std::string key = "\"" + name + "\"";
    const size_t key_pos = blob.find(key, from_pos);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing key in manifest: " + name);
    }
    const size_t colon = blob.find(':', key_pos);
    const size_t begin = blob.find_first_of("-0123456789", colon);
    const size_t end = blob.find_first_not_of("0123456789-", begin);
    if (begin == std::string::npos) {
        throw std::runtime_error("malformed integer key in manifest: " + name);
    }
    return std::stoi(blob.substr(begin, end - begin));
}

inline size_t manifest_find_required(
    const std::string &blob,
    const std::string &needle,
    size_t from_pos = 0
) {
    const size_t pos = blob.find(needle, from_pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing manifest entry: " + needle);
    }
    return pos;
}

inline HardwareManifest load_hardware_manifest(const std::string &export_dir) {
    HardwareManifest out;
    out.export_dir = export_dir;

    std::string path = export_dir + "/hardware_export_manifest.repaired.json";
    {
        std::ifstream repaired(path);
        if (!repaired) {
            path = export_dir + "/hardware_export_manifest.json";
        }
    }
    const std::string blob = manifest_slurp(path);
    out.kv260_status = manifest_extract_string(blob, "kv260_status");

    out.decoder_layers.clear();
    size_t pos = 0;
    for (int layer = 0;; ++layer) {
        const std::string needle = "\"layer\": " + std::to_string(layer);
        const size_t layer_pos = blob.find(needle, pos);
        if (layer_pos == std::string::npos) {
            break;
        }
        WeightOffsets offsets;
        offsets.wgt_bytes = (size_t)manifest_extract_int(blob, "w_offset_bytes", layer_pos);
        offsets.meta_bytes = (size_t)manifest_extract_int(blob, "meta_offset_bytes", layer_pos);
        offsets.wgt_axi = manifest_extract_int(blob, "w_offset_axi256", layer_pos);
        offsets.meta_axi = manifest_extract_int(blob, "meta_offset_axi256", layer_pos);
        out.decoder_layers.push_back(offsets);
        pos = layer_pos + needle.size();
    }

    const size_t vision_pos = manifest_find_required(blob, "\"vision_layers\"");
    const size_t connector_pos = manifest_find_required(blob, "\"connector\"");
    pos = vision_pos;
    while (true) {
        const size_t name_pos = blob.find("\"name\"", pos);
        if (name_pos == std::string::npos || name_pos > connector_pos) {
            break;
        }
        VisionLayerManifest layer;
        layer.name = manifest_extract_string(blob, "name", name_pos);
        layer.wgt_axi = manifest_extract_int(blob, "w_offset_axi256", name_pos);
        layer.meta_axi = manifest_extract_int(blob, "meta_offset_axi256", name_pos);
        layer.M = manifest_extract_int(blob, "M", name_pos);
        layer.K = manifest_extract_int(blob, "K", name_pos);
        layer.N = manifest_extract_int(blob, "N", name_pos);
        out.vision_layers.push_back(layer);
        pos = name_pos + 6;
    }

    const size_t patch_pos = manifest_find_required(blob, "\"name\": \"patch_embed\"", vision_pos);
    out.vision.patch_wgt_axi = manifest_extract_int(blob, "w_offset_axi256", patch_pos);
    out.vision.patch_meta_axi = manifest_extract_int(blob, "meta_offset_axi256", patch_pos);

    const size_t qkv_pos = manifest_find_required(blob, "\"name\": \"blk.0.qkv\"", vision_pos);
    out.vision.qkv_wgt_axi = manifest_extract_int(blob, "w_offset_axi256", qkv_pos);
    out.vision.qkv_meta_axi = manifest_extract_int(blob, "meta_offset_axi256", qkv_pos);

    const size_t attn_pos = manifest_find_required(blob, "\"name\": \"blk.0.attn_out\"", vision_pos);
    out.vision.attn_wgt_axi = manifest_extract_int(blob, "w_offset_axi256", attn_pos);
    out.vision.attn_meta_axi = manifest_extract_int(blob, "meta_offset_axi256", attn_pos);

    out.connector.name = manifest_extract_string(blob, "name", connector_pos);
    out.connector.wgt_axi = manifest_extract_int(blob, "w_offset_axi256", connector_pos);
    out.connector.meta_axi = manifest_extract_int(blob, "meta_offset_axi256", connector_pos);
    out.connector.M = manifest_extract_int(blob, "M", connector_pos);
    out.connector.K = manifest_extract_int(blob, "K", connector_pos);
    out.connector.N = manifest_extract_int(blob, "N", connector_pos);
    out.vision.connector_wgt_axi = out.connector.wgt_axi;
    out.vision.connector_meta_axi = out.connector.meta_axi;

    const char *artifacts[] = {
        "vision_wgt.bin", "vision_meta.bin",
        "connector_wgt.bin", "connector_meta.bin",
        "weights_half0.bin", "weights_half1.bin",
        "meta_half0.bin", "meta_half1.bin",
        "gamma_attn.bin", "gamma_ffn.bin",
        "lmhead_wgt_half0.bin", "lmhead_wgt_half1.bin",
        "lmhead_meta_half0.bin", "lmhead_meta_half1.bin",
    };
    for (const char *name : artifacts) {
        out.artifact_bytes[name] = manifest_file_size(export_dir + "/" + name);
    }
    return out;
}

} // namespace smolvlm2
