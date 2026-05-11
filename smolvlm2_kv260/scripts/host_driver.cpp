// =============================================================================
// host_driver.cpp  —  PS 端 XRT 调度框架（Layer 级 Ping-Pong 流水）
// =============================================================================
// 职责：
//   ① 配置 AXI Lite 寄存器（seq_len, kv_len, pos_start, run_lmhead）
//   ② 每层启动 kernel，轮询完成（ap_done bit0）
//   ③ Layer N 计算期间，预配置 Layer N+1 的 DMA 描述符
//   ④ lm_head 输出后调用 Top-K Sampling（PS 端 C++ 实现）
// 编译：g++ -O2 host_driver.cpp -o smolvlm2_infer -lxrt_coreutil -lxrt_core
// =============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

#include "../llamacpp_fpga/hardware_manifest.h"

// XRT 头文件（需要 XRT 安装，路径 /opt/xilinx/xrt/include）
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

// 模型常数（与 HLS 端保持同步）
static constexpr int C_DIM       = 960;
static constexpr int H_KV        = 5;
static constexpr int D_HEAD      = 64;
static constexpr int FFN_DIM     = 2560;
static constexpr int VOCAB_SIZE  = 49280;
static constexpr int NUM_LAYERS  = 32;
static constexpr int MODEL_MAX_SEQ = 8192;
static constexpr int HW_MAX_SEQ    = 2048;
static constexpr int MAX_SEQ     = HW_MAX_SEQ;
static constexpr int GRP         = 32;
static constexpr int VIT_TOKENS  = 1024;
static constexpr int VIT_C       = 768;
static constexpr int VIT_QKV     = 2304;
static constexpr int VIT_FFN     = 3072;
static constexpr int IMAGE_TOKENS = 64;
static constexpr int CONNECTOR_IN = 12288;
static constexpr int CONNECTOR_OUT = 960;
static constexpr int IMAGE_TOKEN_ID = 49190;

// AXI256 = 32 byte
static constexpr int AXI_W       = 32;

using smolvlm2::HardwareManifest;
using smolvlm2::VisionOffsets;
using smolvlm2::WeightOffsets;

// ---------------------------------------------------------------------------
// 每层权重大小（INT4 字节数）
// QKV(960×1600) + O(960×960) + FFN_gate_up(960×5120) + FFN_down(2560×960)
// ---------------------------------------------------------------------------
static constexpr size_t QKV_BYTES    = (size_t)C_DIM * (C_DIM + H_KV*D_HEAD*2) / 2;
static constexpr size_t OPROJ_BYTES  = (size_t)C_DIM * C_DIM / 2;
static constexpr size_t GATEUP_BYTES = (size_t)C_DIM * FFN_DIM * 2 / 2;
static constexpr size_t DOWN_BYTES   = (size_t)FFN_DIM * C_DIM / 2;
static constexpr size_t LAYER_WGT_BYTES = QKV_BYTES + OPROJ_BYTES + GATEUP_BYTES + DOWN_BYTES;

// meta 字节数（每 group 2 byte）
static constexpr size_t LAYER_META_BYTES = LAYER_WGT_BYTES / GRP * 2;

// ---------------------------------------------------------------------------
// KV Cache 大小（INT4，所有层）
// 32层 × 5头 × 2(K+V) × 2000seq × 64dim × 0.5byte ≈ 40 MB
// ---------------------------------------------------------------------------
static constexpr size_t KV_CACHE_TOTAL = (size_t)NUM_LAYERS * H_KV * 2 * MAX_SEQ * D_HEAD / 2;

// ---------------------------------------------------------------------------
// SmolVLM2Inferencer
// ---------------------------------------------------------------------------
class SmolVLM2Inferencer {
public:
    SmolVLM2Inferencer(const std::string &xclbin_path,
                        const std::string &export_dir)
    {
        // XRT 设备初始化
        device_ = xrt::device(0);  // KV260 = device 0
        auto uuid = device_.load_xclbin(xclbin_path);
        kernel_  = xrt::kernel(device_, uuid, "smolvlm2_decoder_layer",
                               xrt::kernel::cu_access_mode::exclusive);
        vit_kernel_ = xrt::kernel(device_, uuid, "smolvlm2_vit_prefill_kernel",
                                  xrt::kernel::cu_access_mode::exclusive);
        connector_kernel_ = xrt::kernel(device_, uuid, "smolvlm2_connector_kernel",
                                        xrt::kernel::cu_access_mode::exclusive);
        bridge_kernel_ = xrt::kernel(device_, uuid, "smolvlm2_image_to_decoder_bridge_kernel",
                                     xrt::kernel::cu_access_mode::exclusive);

        std::cout << "[HOST] XRT kernel loaded: smolvlm2_decoder_layer\n";
        std::cout << "[HOST] Sequence capability: model_max_seq=" << MODEL_MAX_SEQ
                  << " hw_max_seq=" << HW_MAX_SEQ << " mode=restricted\n";

        export_dir_ = export_dir;

        // 分配 DDR 缓冲（XRT Buffer Object）
        alloc_buffers();

        load_manifest();
        load_weights();

        std::cout << "[HOST] Buffers allocated, weights loaded.\n";
    }

    // -----------------------------------------------------------------------
    // Prefill：处理 prompt token 序列
    // -----------------------------------------------------------------------
    std::vector<int> prefill(const std::vector<int> &token_ids,
                              int max_new_tokens = 128)
    {
        int L = token_ids.size();
        std::cout << "[HOST] Prefill: " << L << " tokens\n";

        encode_image();

        // 将 image tokens 和 text tokens 拼成 decoder prefill 输入。
        // 当前结构先经 DDR 桥接，后续可以替换为 PL 端 connector->decoder pack bridge。
        L = build_prefill_activations(token_ids, act_ping_bo_);

        run_prefill_throughput(L);

        // Decode 阶段
        return decode(L, max_new_tokens);
    }

private:
    static constexpr int PREFILL_TOKEN_TILE = 32;
    static constexpr int DECODE_WINDOW = 2;
    static constexpr int EXEC_MODE_TTFT_DECODE = 0;
    static constexpr int EXEC_MODE_THROUGHPUT_PREFILL = 1;
    static constexpr int EXEC_MODE_THROUGHPUT_DECODE = 2;

    xrt::device device_;
    xrt::kernel  kernel_;
    xrt::kernel  vit_kernel_;
    xrt::kernel  connector_kernel_;
    xrt::kernel  bridge_kernel_;
    std::string export_dir_;

    // XRT Buffer Objects（DDR 分配）
    xrt::bo act_ping_bo_, act_pong_bo_, act_out_bo_;
    xrt::bo vit_act_bo_, vit_hidden_bo_, vit_tmp_bo_;
    xrt::bo vision_wgt_bo_, vision_meta_bo_;
    xrt::bo connector_wgt_bo_, connector_meta_bo_, image_tokens_bo_;
    xrt::bo wgt_half0_bo_, wgt_half1_bo_;
    xrt::bo meta_half0_bo_, meta_half1_bo_;
    xrt::bo kv_k_bo_, kv_v_bo_;
    xrt::bo gamma_attn_bo_, gamma_ffn_bo_;
    xrt::bo lmhead_wgt_bo_, lmhead_meta_bo_, lmhead_out_bo_;

    // 权重 DDR 偏移（每层不同）
    size_t cur_wgt_offset_      = 0;
    size_t cur_meta_offset_     = 0;
    int    cur_wgt_axi_offset_  = 0;
    int    cur_meta_axi_offset_ = 0;
    HardwareManifest manifest_;

    void alloc_buffers() {
        size_t act_bytes    = MAX_SEQ * C_DIM;          // INT8
        size_t vit_act_bytes = VIT_TOKENS * VIT_C;
        size_t vit_hidden_bytes = VIT_TOKENS * VIT_C * sizeof(int32_t);
        size_t vit_tmp_bytes = VIT_TOKENS * VIT_FFN * sizeof(int32_t);
        size_t vision_wgt_bytes = 128 * 1024 * 1024;
        size_t vision_meta_bytes = 16 * 1024 * 1024;
        size_t connector_wgt_bytes = (size_t)CONNECTOR_IN * CONNECTOR_OUT / 2;
        size_t connector_meta_bytes = (size_t)CONNECTOR_IN * CONNECTOR_OUT / GRP * 2;
        size_t image_token_bytes = IMAGE_TOKENS * CONNECTOR_OUT * sizeof(int32_t);
        size_t wgt_bytes    = NUM_LAYERS * LAYER_WGT_BYTES;
        size_t meta_bytes   = NUM_LAYERS * LAYER_META_BYTES;
        size_t kv_bytes     = KV_CACHE_TOTAL;
        size_t gamma_bytes  = C_DIM;                    // INT8，单层（广播复用）
        size_t lm_wgt_bytes = (size_t)C_DIM * VOCAB_SIZE / 2;
        size_t lm_meta_bytes= (size_t)C_DIM * VOCAB_SIZE / GRP * 2;
        size_t logit_bytes  = VOCAB_SIZE * 4;           // INT32

        vit_act_bo_ = xrt::bo(device_, vit_act_bytes, XCL_BO_FLAGS_NONE, vit_kernel_.group_id(0));
        vit_hidden_bo_ = xrt::bo(device_, vit_hidden_bytes, XCL_BO_FLAGS_NONE, vit_kernel_.group_id(3));
        vit_tmp_bo_ = xrt::bo(device_, vit_tmp_bytes, XCL_BO_FLAGS_NONE, vit_kernel_.group_id(3));
        vision_wgt_bo_ = xrt::bo(device_, vision_wgt_bytes, XCL_BO_FLAGS_NONE, vit_kernel_.group_id(1));
        vision_meta_bo_ = xrt::bo(device_, vision_meta_bytes, XCL_BO_FLAGS_NONE, vit_kernel_.group_id(2));
        connector_wgt_bo_ = xrt::bo(device_, connector_wgt_bytes, XCL_BO_FLAGS_NONE, connector_kernel_.group_id(1));
        connector_meta_bo_ = xrt::bo(device_, connector_meta_bytes, XCL_BO_FLAGS_NONE, connector_kernel_.group_id(2));
        image_tokens_bo_ = xrt::bo(device_, image_token_bytes, XCL_BO_FLAGS_NONE, connector_kernel_.group_id(3));

        // HPC0 group
        act_ping_bo_  = xrt::bo(device_, act_bytes,   XCL_BO_FLAGS_NONE, kernel_.group_id(0));
        act_pong_bo_  = xrt::bo(device_, act_bytes,   XCL_BO_FLAGS_NONE, kernel_.group_id(1));
        act_out_bo_   = xrt::bo(device_, act_bytes,   XCL_BO_FLAGS_NONE, kernel_.group_id(2));
        wgt_half0_bo_ = xrt::bo(device_, wgt_bytes/2, XCL_BO_FLAGS_NONE, kernel_.group_id(3));

        // HPC1 group
        wgt_half1_bo_   = xrt::bo(device_, wgt_bytes/2,   XCL_BO_FLAGS_NONE, kernel_.group_id(4));
        meta_half0_bo_  = xrt::bo(device_, meta_bytes/2,  XCL_BO_FLAGS_NONE, kernel_.group_id(5));
        meta_half1_bo_  = xrt::bo(device_, meta_bytes/2,  XCL_BO_FLAGS_NONE, kernel_.group_id(6));
        kv_k_bo_        = xrt::bo(device_, kv_bytes/2,    XCL_BO_FLAGS_NONE, kernel_.group_id(7));
        kv_v_bo_        = xrt::bo(device_, kv_bytes/2,    XCL_BO_FLAGS_NONE, kernel_.group_id(8));
        gamma_attn_bo_  = xrt::bo(device_, gamma_bytes,   XCL_BO_FLAGS_NONE, kernel_.group_id(9));
        gamma_ffn_bo_   = xrt::bo(device_, gamma_bytes,   XCL_BO_FLAGS_NONE, kernel_.group_id(10));
        lmhead_wgt_bo_  = xrt::bo(device_, lm_wgt_bytes,  XCL_BO_FLAGS_NONE, kernel_.group_id(11));
        lmhead_meta_bo_ = xrt::bo(device_, lm_meta_bytes, XCL_BO_FLAGS_NONE, kernel_.group_id(12));
        lmhead_out_bo_  = xrt::bo(device_, logit_bytes,   XCL_BO_FLAGS_NONE, kernel_.group_id(13));

        std::cout << "[HOST] Buffers allocated: "
                  << "wgt=" << (wgt_bytes/1024/1024) << "MB "
                  << "KV="  << (kv_bytes/1024/1024)  << "MB\n";
    }

    void load_manifest() {
        manifest_ = smolvlm2::load_hardware_manifest(export_dir_);
        if (manifest_.kv260_status != "native_w4a8_ready_host_folded_awq") {
            throw std::runtime_error("unexpected kv260_status: " + manifest_.kv260_status);
        }
        if ((int)manifest_.decoder_layers.size() != NUM_LAYERS) {
            throw std::runtime_error("decoder layer count mismatch in manifest");
        }
        std::cout << "[HOST] Manifest loaded: decoder_layers="
                  << manifest_.decoder_layers.size()
                  << " vision_layers=" << manifest_.vision_layers.size()
                  << " connector=" << manifest_.connector.name << "\n";
    }

    void enforce_hw_seq_limit(int total_tokens) const {
        if (total_tokens > HW_MAX_SEQ) {
            throw std::runtime_error(
                "requested token count " + std::to_string(total_tokens) +
                " exceeds hw_max_seq=" + std::to_string(HW_MAX_SEQ) +
                " while model_max_seq=" + std::to_string(MODEL_MAX_SEQ)
            );
        }
    }

    void load_binary_to_bo(const std::string &path, xrt::bo &bo) {
        FILE *fp = fopen(path.c_str(), "rb");
        if (!fp) {
            std::cerr << "[HOST] WARNING: file not found: " << path << "\n";
            return;
        }
        auto *dst = bo.map<uint8_t *>();
        const size_t bytes = fread(dst, 1, bo.size(), fp);
        fclose(fp);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::cout << "[HOST] Loaded " << bytes << " bytes from " << path << "\n";
    }

    void load_pair_to_single_bo(const std::string &path0, const std::string &path1, xrt::bo &bo) {
        auto *dst = bo.map<uint8_t *>();
        memset(dst, 0, bo.size());

        size_t offset = 0;
        for (const auto &path : {path0, path1}) {
            FILE *fp = fopen(path.c_str(), "rb");
            if (!fp) {
                std::cerr << "[HOST] WARNING: file not found: " << path << "\n";
                continue;
            }
            const size_t bytes = fread(dst + offset, 1, bo.size() - offset, fp);
            fclose(fp);
            std::cout << "[HOST] Loaded " << bytes << " bytes from " << path << " @offset " << offset << "\n";
            offset += bytes;
        }
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void load_weights() {
        load_binary_to_bo(export_dir_ + "/vision_wgt.bin",    vision_wgt_bo_);
        load_binary_to_bo(export_dir_ + "/vision_meta.bin",   vision_meta_bo_);
        load_binary_to_bo(export_dir_ + "/connector_wgt.bin", connector_wgt_bo_);
        load_binary_to_bo(export_dir_ + "/connector_meta.bin", connector_meta_bo_);
        load_binary_to_bo(export_dir_ + "/weights_half0.bin",  wgt_half0_bo_);
        load_binary_to_bo(export_dir_ + "/weights_half1.bin",  wgt_half1_bo_);
        load_binary_to_bo(export_dir_ + "/meta_half0.bin",     meta_half0_bo_);
        load_binary_to_bo(export_dir_ + "/meta_half1.bin",     meta_half1_bo_);
        load_binary_to_bo(export_dir_ + "/gamma_attn.bin",     gamma_attn_bo_);
        load_binary_to_bo(export_dir_ + "/gamma_ffn.bin",      gamma_ffn_bo_);
        load_pair_to_single_bo(
            export_dir_ + "/lmhead_wgt_half0.bin",
            export_dir_ + "/lmhead_wgt_half1.bin",
            lmhead_wgt_bo_
        );
        load_pair_to_single_bo(
            export_dir_ + "/lmhead_meta_half0.bin",
            export_dir_ + "/lmhead_meta_half1.bin",
            lmhead_meta_bo_
        );
    }

    int build_prefill_activations(const std::vector<int> &token_ids, xrt::bo &act_bo) {
        auto *act_ptr = act_bo.map<int8_t*>();
        memset(act_ptr, 0, MAX_SEQ * C_DIM);

        int out_token = 0;
        bool saw_image_placeholder = false;
        int image_dst_token = 0;
        auto reserve_image_tokens = [&]() {
            if (out_token + IMAGE_TOKENS > MAX_SEQ) {
                throw std::runtime_error("prefill image tokens exceed MAX_SEQ");
            }
            image_dst_token = out_token;
            out_token += IMAGE_TOKENS;
        };

        for (int token_id : token_ids) {
            if (token_id == IMAGE_TOKEN_ID) {
                reserve_image_tokens();
                saw_image_placeholder = true;
                continue;
            }
            if (out_token >= MAX_SEQ) {
                throw std::runtime_error("prefill text tokens exceed MAX_SEQ");
            }
            // Placeholder: text embedding lookup is still a PS-side integration item.
            // Keeping zeros here makes the image/decoder data path structurally testable.
            out_token += 1;
        }
        if (!saw_image_placeholder) {
            out_token = 0;
            reserve_image_tokens();
            if (out_token + (int)token_ids.size() > MAX_SEQ) {
                throw std::runtime_error("prefill tokens exceed MAX_SEQ after prepending image tokens");
            }
            out_token += (int)token_ids.size();
        }

        enforce_hw_seq_limit(out_token);
        act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        run_image_bridge(image_dst_token, IMAGE_TOKENS);
        std::cout << "[HOST] Prefill activations built: input_tokens="
                  << token_ids.size() << " decoder_tokens=" << out_token
                  << " image_placeholder=" << (saw_image_placeholder ? "yes" : "prepended") << "\n";
        return out_token;
    }

    void run_image_bridge(int dst_token_offset, int num_image_tokens) {
        auto bridge = bridge_kernel_(
            image_tokens_bo_,
            act_ping_bo_,
            dst_token_offset,
            num_image_tokens,
            C_DIM
        );
        bridge.wait();
    }

    void encode_image() {
        const VisionOffsets &vision_offsets_ = manifest_.vision;
        auto patch = vit_kernel_(
            vit_act_bo_,
            vision_wgt_bo_,
            vision_meta_bo_,
            vit_hidden_bo_,
            1,
            VIT_TOKENS,
            VIT_C,
            VIT_C,
            vision_offsets_.patch_wgt_axi,
            vision_offsets_.patch_meta_axi
        );
        patch.wait();

        auto qkv = vit_kernel_(
            vit_hidden_bo_,
            vision_wgt_bo_,
            vision_meta_bo_,
            vit_tmp_bo_,
            2,
            VIT_TOKENS,
            VIT_C,
            VIT_QKV,
            vision_offsets_.qkv_wgt_axi,
            vision_offsets_.qkv_meta_axi
        );
        qkv.wait();

        auto attn = vit_kernel_(
            vit_tmp_bo_,
            vision_wgt_bo_,
            vision_meta_bo_,
            vit_hidden_bo_,
            3,
            VIT_TOKENS,
            VIT_QKV,
            VIT_C,
            vision_offsets_.attn_wgt_axi,
            vision_offsets_.attn_meta_axi
        );
        attn.wait();

        auto conn = connector_kernel_(
            vit_hidden_bo_,
            connector_wgt_bo_,
            connector_meta_bo_,
            image_tokens_bo_,
            1,
            vision_offsets_.connector_wgt_axi,
            vision_offsets_.connector_meta_axi
        );
        conn.wait();
    }

    void setup_weight_ptrs(int layer) {
        if (layer < 0 || layer >= (int)manifest_.decoder_layers.size()) {
            throw std::runtime_error("decoder layer offset missing in manifest for layer " + std::to_string(layer));
        }
        cur_wgt_axi_offset_  = manifest_.decoder_layers[layer].wgt_axi;
        cur_meta_axi_offset_ = manifest_.decoder_layers[layer].meta_axi;
        cur_wgt_offset_  = (size_t)cur_wgt_axi_offset_ * AXI_W;
        cur_meta_offset_ = (size_t)cur_meta_axi_offset_ * AXI_W;
    }

    void launch_decoder_stage(
        int seq_len,
        int kv_len,
        int pos_start,
        int exec_mode,
        int token_tile_offset,
        int token_tile_size,
        int run_lmhead,
        int layer,
        int use_ping
    ) {
        auto run = kernel_(
            use_ping ? act_ping_bo_ : act_pong_bo_,
            use_ping ? act_pong_bo_ : act_ping_bo_,
            use_ping ? act_pong_bo_ : act_ping_bo_,
            wgt_half0_bo_, wgt_half1_bo_,
            meta_half0_bo_, meta_half1_bo_,
            kv_k_bo_, kv_v_bo_,
            gamma_attn_bo_, gamma_ffn_bo_,
            lmhead_wgt_bo_, lmhead_meta_bo_, lmhead_out_bo_,
            seq_len,
            kv_len,
            pos_start,
            exec_mode,
            token_tile_offset,
            token_tile_size,
            run_lmhead,
            layer,
            use_ping,
            cur_wgt_axi_offset_,
            cur_meta_axi_offset_,
            1,
            H_KV * 3,
            H_KV
        );
        run.wait();
    }

    void run_prefill_throughput(int seq_len) {
        std::cout << "[HOST] Path=throughput_prefill tile=" << PREFILL_TOKEN_TILE
                  << " seq_len=" << seq_len << "\n";
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            setup_weight_ptrs(layer);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int tile_offset = 0; tile_offset < seq_len; tile_offset += PREFILL_TOKEN_TILE) {
                const int tile_size = std::min(PREFILL_TOKEN_TILE, seq_len - tile_offset);
                bool use_ping = (layer % 2 == 0);
                launch_decoder_stage(
                    seq_len, seq_len, 0,
                    EXEC_MODE_THROUGHPUT_PREFILL,
                    tile_offset, tile_size,
                    0, layer, use_ping ? 1 : 0
                );
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "[HOST] Prefill layer " << layer << " tiles done: " << ms << " ms\n";
        }
    }

    void run_decode_ttft_token(int pos, int kv_len) {
        std::cout << "[HOST] Path=ttft_decode pos=" << pos << " kv_len=" << kv_len << "\n";
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            bool use_ping = (layer % 2 == 0);
            int run_lmh = (layer == NUM_LAYERS - 1) ? 1 : 0;
            setup_weight_ptrs(layer);
            launch_decoder_stage(
                1, kv_len + 1, pos,
                EXEC_MODE_TTFT_DECODE,
                0, 1,
                run_lmh, layer, use_ping ? 1 : 0
            );
        }
    }

    void run_decode_throughput_window(int pos, int kv_len, int window_tokens) {
        std::cout << "[HOST] Path=throughput_decode window=" << window_tokens
                  << " pos=" << pos << " kv_len=" << kv_len << "\n";
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            bool use_ping = (layer % 2 == 0);
            int run_lmh = (layer == NUM_LAYERS - 1) ? 1 : 0;
            setup_weight_ptrs(layer);
            launch_decoder_stage(
                window_tokens, kv_len + window_tokens, pos,
                EXEC_MODE_THROUGHPUT_DECODE,
                0, window_tokens,
                run_lmh, layer, use_ping ? 1 : 0
            );
        }
    }

    std::vector<int> decode(int prompt_len, int max_new_tokens) {
        std::vector<int> generated_ids;
        int cur_pos  = prompt_len;
        int cur_kv   = prompt_len;

        std::cout << "[HOST] Decode: generating up to " << max_new_tokens << " tokens\n";

        for (int step = 0; step < max_new_tokens; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();

            if (step == 0) {
                run_decode_ttft_token(cur_pos, cur_kv);
            } else {
                run_decode_throughput_window(cur_pos, cur_kv, 1);
            }

            // 读取 logits，做 Top-K Sampling（PS 端）
            lmhead_out_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto *logits = lmhead_out_bo_.map<int32_t*>();

            int next_token = top1_sampling(logits, VOCAB_SIZE);
            generated_ids.push_back(next_token);

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
            std::cout << "[HOST] Step " << step << " token=" << next_token
                      << " (" << ms << " ms/token)\n";

            // EOS 检测（token_id=2 通常为 EOS）
            if (next_token == 2) break;

            // 准备下一步：嵌入 next_token 写入 act_ping
            // （省略 embedding lookup，实际需要查嵌入表）

            cur_pos++;
            cur_kv++;
        }

        return generated_ids;
    }

    // Top-1（greedy）采样（Top-K 采样扩展：实际工程需要 softmax + multinomial）
    int top1_sampling(const int32_t *logits, int vocab_size) {
        int best_id  = 0;
        int32_t best = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > best) {
                best    = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <xclbin_path> <hardware_export_dir> [prompt_text]\n";
        return 1;
    }

    std::string xclbin  = argv[1];
    std::string export_dir = argv[2];
    std::string prompt  = (argc >= 4) ? argv[3] : "Hello";

    std::cout << "=== SmolVLM2-500M on KV260 ===\n";
    std::cout << "XCLBIN:  " << xclbin  << "\n";
    std::cout << "Export:  " << export_dir << "\n";
    std::cout << "Prompt:  " << prompt  << "\n\n";

    try {
        SmolVLM2Inferencer infer(xclbin, export_dir);

        // 简化：假设 tokenizer 已将 prompt 转为 token ids
        // 实际需要集成 tokenizer（如 sentencepiece）
        std::vector<int> token_ids = {1, 100, 200, 300};  // placeholder

        auto output_ids = infer.prefill(token_ids, 128);

        std::cout << "\n[RESULT] Generated token IDs:";
        for (int id : output_ids) std::cout << " " << id;
        std::cout << "\n";

    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
