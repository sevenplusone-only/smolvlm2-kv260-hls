// =============================================================================
// vit_kernel.h  —  SmolVLM2 vision shape-matched HLS kernel declarations
// =============================================================================
#pragma once
#include "common.h"

extern "C" void smolvlm2_vit_prefill_kernel(
    const AXI256 *act_in,
    const AXI256 *wgt,
    const AXI256 *meta,
          AXI256 *act_out,
    int           mode,
    int           M,
    int           K,
    int           N,
    int           wgt_offset,
    int           meta_offset
);

extern "C" void smolvlm2_connector_kernel(
    const AXI256 *vit_tokens,
    const AXI256 *projector_wgt,
    const AXI256 *projector_meta,
          AXI256 *image_tokens,
    int           run_projector,
    int           wgt_offset,
    int           meta_offset
);
