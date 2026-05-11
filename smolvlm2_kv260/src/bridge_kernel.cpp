// =============================================================================
// bridge_kernel.cpp  —  Device-side bridge from connector INT32 image tokens
//                        to decoder INT8 activation buffer.
// =============================================================================
#include "common.h"

extern "C" void smolvlm2_image_to_decoder_bridge_kernel(
    const AXI256 *image_tokens,
          AXI256 *decoder_act,
    int           dst_token_offset,
    int           num_image_tokens,
    int           hidden_size
) {
#pragma HLS INTERFACE m_axi port=image_tokens bundle=gmem_img offset=slave max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=decoder_act  bundle=gmem_act offset=slave max_read_burst_length=16 max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=dst_token_offset bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_image_tokens bundle=ctrl
#pragma HLS INTERFACE s_axilite port=hidden_size bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    if (hidden_size != C || num_image_tokens <= 0) {
        return;
    }

    const int total_out_words = (num_image_tokens * hidden_size) / 32;
    const int dst_word_base = (dst_token_offset * hidden_size) / 32;

    for (int ow = 0; ow < total_out_words; ++ow) {
#pragma HLS PIPELINE II=1
        AXI256 dst = 0;
        for (int group = 0; group < 4; ++group) {
#pragma HLS UNROLL
            AXI256 src = image_tokens[ow * 4 + group];
            for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
                INT32 v = (INT32)src.range(lane * 32 + 31, lane * 32);
                INT8 q = (INT8)(v >> 8);
                const int byte_lane = group * 8 + lane;
                dst.range(byte_lane * 8 + 7, byte_lane * 8) = (ap_uint<8>)q;
            }
        }

        decoder_act[dst_word_base + ow] = dst;
    }
}
