# =============================================================================
# hls_synth_bridge.tcl  —  Vitis HLS script for connector->decoder bridge
# =============================================================================
set project_name  "smolvlm2_bridge"
set solution_name "sol_200mhz"
set part          "xck26-sfvc784-2LV-c"
set clock_period  "5ns"

if {[file exists $project_name]} {
    file delete -force $project_name
}

open_project $project_name
set_top smolvlm2_image_to_decoder_bridge_kernel

add_files ../src/bridge_kernel.cpp \
    -cflags "-I../src -O2 -std=c++14"

open_solution $solution_name -flow_target vitis
set_part $part
create_clock -period $clock_period -name default

config_interface -m_axi_addr64
config_compile -pipeline_style flp

csynth_design

export_design -flow syn -rtl verilog \
    -format xo \
    -output "${project_name}_kernel.xo"

puts "============================================================"
puts "  Bridge HLS synthesis complete."
puts "  XO file: ${project_name}_kernel.xo"
puts "============================================================"
close_project
