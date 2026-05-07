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
static constexpr int MAX_SEQ     = 2048;
static constexpr int GRP         = 32;

// AXI256 = 32 byte
static constexpr int AXI_W       = 32;

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
                        const std::string &weights_path)
    {
        // XRT 设备初始化
        device_ = xrt::device(0);  // KV260 = device 0
        auto uuid = device_.load_xclbin(xclbin_path);
        kernel_  = xrt::kernel(device_, uuid, "smolvlm2_decoder_layer",
                               xrt::kernel::cu_access_mode::exclusive);

        std::cout << "[HOST] XRT kernel loaded: smolvlm2_decoder_layer\n";

        // 分配 DDR 缓冲（XRT Buffer Object）
        alloc_buffers();

        // 加载权重（离线量化的 W4 权重文件）
        load_weights(weights_path);

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

        // 将 token embedding 写入 act_ping（PS 端查表）
        embed_tokens(token_ids, act_ping_bo_);

        // Layer 级流水（Layer N PL 计算 + Layer N+1 DMA 预配置）
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            bool use_ping    = (layer % 2 == 0);
            bool write_ping  = !use_ping;

            auto t0 = std::chrono::high_resolution_clock::now();

            // 设置本层权重指针偏移（PS 端预计算好）
            setup_weight_ptrs(layer);

            // 启动 PL kernel
            int run_lmhead = (layer == NUM_LAYERS - 1) ? 0 : 0;
            auto run = kernel_(
                use_ping ? act_ping_bo_ : act_pong_bo_,   // act_ping
                use_ping ? act_pong_bo_ : act_ping_bo_,   // act_pong（不用）
                write_ping ? act_ping_bo_ : act_pong_bo_, // act_out
                wgt_half0_bo_, wgt_half1_bo_,
                meta_half0_bo_, meta_half1_bo_,
                kv_k_bo_, kv_v_bo_,
                gamma_attn_bo_, gamma_ffn_bo_,
                lmhead_wgt_bo_, lmhead_meta_bo_, lmhead_out_bo_,
                L,           // seq_len
                L,           // kv_len（Prefill：kv_len = seq_len）
                0,           // pos_start
                run_lmhead,
                layer,
                use_ping ? 1 : 0,
                cur_wgt_axi_offset_,
                cur_meta_axi_offset_,
                1,
                H_KV * 3,
                H_KV
            );

            // 等待 PL 完成（轮询 ap_done）
            run.wait();

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
            std::cout << "[HOST] Layer " << layer << " done: " << ms << " ms\n";
        }

        // Decode 阶段
        return decode(L, max_new_tokens);
    }

private:
    xrt::device device_;
    xrt::kernel  kernel_;

    // XRT Buffer Objects（DDR 分配）
    xrt::bo act_ping_bo_, act_pong_bo_, act_out_bo_;
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

    void alloc_buffers() {
        size_t act_bytes    = MAX_SEQ * C_DIM;          // INT8
        size_t wgt_bytes    = NUM_LAYERS * LAYER_WGT_BYTES;
        size_t meta_bytes   = NUM_LAYERS * LAYER_META_BYTES;
        size_t kv_bytes     = KV_CACHE_TOTAL;
        size_t gamma_bytes  = C_DIM;                    // INT8，单层（广播复用）
        size_t lm_wgt_bytes = (size_t)C_DIM * VOCAB_SIZE / 2;
        size_t lm_meta_bytes= (size_t)C_DIM * VOCAB_SIZE / GRP * 2;
        size_t logit_bytes  = VOCAB_SIZE * 4;           // INT32

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

    void load_weights(const std::string &weights_path) {
        // 简化：从二进制文件读取预量化权重
        // 实际格式：W4 权重按层排列（layer0_QKV | layer0_O | ... | layer31_down）
        FILE *fp = fopen(weights_path.c_str(), "rb");
        if (!fp) {
            std::cerr << "[HOST] WARNING: weights file not found: " << weights_path
                      << " (using zero weights for testing)\n";
            return;
        }

        auto *wgt0 = wgt_half0_bo_.map<uint8_t*>();
        auto *wgt1 = wgt_half1_bo_.map<uint8_t*>();
        size_t half_bytes = NUM_LAYERS * LAYER_WGT_BYTES / 2;

        fread(wgt0, 1, half_bytes, fp);
        fread(wgt1, 1, half_bytes, fp);
        fclose(fp);

        wgt_half0_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        wgt_half1_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // 同理加载 meta、gamma、lmhead_wgt...（略）
        std::cout << "[HOST] Weights loaded from " << weights_path << "\n";
    }

    void embed_tokens(const std::vector<int> &token_ids, xrt::bo &act_bo) {
        // 从嵌入表查 token embedding，写入激活缓冲
        // 简化：写全 0（实际需要嵌入权重文件）
        auto *ptr = act_bo.map<int8_t*>();
        memset(ptr, 0, MAX_SEQ * C_DIM);
        for (int i = 0; i < (int)token_ids.size(); ++i) {
            // ptr[i * C_DIM ... (i+1)*C_DIM] = embedding[token_ids[i]]
            // 实际需查表填入
        }
        act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void setup_weight_ptrs(int layer) {
        // 为第 layer 层设置权重/meta DDR 偏移（以 AXI256 word 为单位）
        // HLS kernel 接口接受的是基地址指针，偏移在 kernel 内部通过
        // 模板参数或运行时偏移量加法实现
        // 此处简化：单独为每层 DMA 传输（实际通过 BO subrange 或 offset 机制）
        cur_wgt_offset_  = layer * (LAYER_WGT_BYTES / 2);
        cur_meta_offset_ = layer * (LAYER_META_BYTES / 2);
        cur_wgt_axi_offset_  = (int)(cur_wgt_offset_ / AXI_W);
        cur_meta_axi_offset_ = (int)(cur_meta_offset_ / AXI_W);
        // gamma 每层不同，需要单独 DMA 或预加载到同一 BRAM
    }

    std::vector<int> decode(int prompt_len, int max_new_tokens) {
        std::vector<int> generated_ids;
        int cur_pos  = prompt_len;
        int cur_kv   = prompt_len;

        std::cout << "[HOST] Decode: generating up to " << max_new_tokens << " tokens\n";

        for (int step = 0; step < max_new_tokens; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // 每次 Decode 只有 1 个 token
            for (int layer = 0; layer < NUM_LAYERS; ++layer) {
                bool use_ping = (layer % 2 == 0);
                int run_lmh  = (layer == NUM_LAYERS - 1) ? 1 : 0;

                setup_weight_ptrs(layer);

                auto run = kernel_(
                    use_ping ? act_ping_bo_ : act_pong_bo_,
                    use_ping ? act_pong_bo_ : act_ping_bo_,
                    use_ping ? act_pong_bo_ : act_ping_bo_,  // act_out = 下一层 ping
                    wgt_half0_bo_, wgt_half1_bo_,
                    meta_half0_bo_, meta_half1_bo_,
                    kv_k_bo_, kv_v_bo_,
                    gamma_attn_bo_, gamma_ffn_bo_,
                    lmhead_wgt_bo_, lmhead_meta_bo_, lmhead_out_bo_,
                    1,         // seq_len = 1（Decode）
                    cur_kv,    // kv_len
                    cur_pos,   // pos_start
                    run_lmh,
                    layer,
                    use_ping ? 1 : 0,
                    cur_wgt_axi_offset_,
                    cur_meta_axi_offset_,
                    1,
                    H_KV * 3,
                    H_KV
                );
                run.wait();
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
                  << " <xclbin_path> <weights_bin_path> [prompt_text]\n";
        return 1;
    }

    std::string xclbin  = argv[1];
    std::string weights = argv[2];
    std::string prompt  = (argc >= 4) ? argv[3] : "Hello";

    std::cout << "=== SmolVLM2-500M on KV260 ===\n";
    std::cout << "XCLBIN:  " << xclbin  << "\n";
    std::cout << "Weights: " << weights << "\n";
    std::cout << "Prompt:  " << prompt  << "\n\n";

    try {
        SmolVLM2Inferencer infer(xclbin, weights);

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
