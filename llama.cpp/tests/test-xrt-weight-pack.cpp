#include "../ggml/src/ggml-xrt/weight-pack.h"
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cstdlib>

int main() {
    const size_t N = 4, K = 64;
    std::vector<float> src_f32(N*K);
    for (size_t i = 0; i < N*K; ++i) src_f32[i] = ((int)i % 17 - 8) * 0.1f;

    // Use ggml's native quantize_row_q8_0 to produce golden block_q8_0 layout
    std::vector<uint8_t> gguf(N * (K/32) * sizeof(block_q8_0));
    for (size_t n = 0; n < N; ++n) {
        quantize_row_q8_0_ref(&src_f32[n*K], (block_q8_0 *)(gguf.data() + n*(K/32)*sizeof(block_q8_0)), K);
    }

    // Unpack via our function
    std::vector<int8_t>  int8_out(N*K);
    std::vector<uint16_t> fp16_out(N*(K/32));
    ggml_xrt::unpack_gguf_q8_to_split(gguf.data(), int8_out.data(), fp16_out.data(), N, K);

    // Verify: for each block, the int8 quants match and fp16 scale matches
    int err = 0;
    for (size_t n = 0; n < N; ++n) {
        for (size_t kg = 0; kg < K/32; ++kg) {
            const block_q8_0 * blk = (const block_q8_0 *)(gguf.data() + (n*(K/32)+kg)*sizeof(block_q8_0));
            if (*(uint16_t*)&blk->d != fp16_out[n*(K/32) + kg]) {
                printf("FAIL scale n=%zu kg=%zu expect %04x got %04x\n",
                    n, kg, *(uint16_t*)&blk->d, fp16_out[n*(K/32)+kg]);
                ++err;
            }
            for (int i = 0; i < 32; ++i) {
                if (blk->qs[i] != int8_out[n*K + kg*32 + i]) {
                    if (err < 8) printf("FAIL q n=%zu kg=%zu i=%d expect %d got %d\n",
                        n, kg, i, blk->qs[i], int8_out[n*K+kg*32+i]);
                    ++err;
                }
            }
        }
    }
    printf("errors = %d / %zu\n", err, N*K + N*(K/32));
    return err ? 1 : 0;
}
