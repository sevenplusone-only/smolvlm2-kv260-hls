# =============================================================================
# hls_synth.tcl  —  Vitis HLS 综合脚本
# 目标：xck26（KV260），200 MHz，单 kernel smolvlm2_decoder_layer
# 用法：vitis_hls -f hls_synth.tcl
# =============================================================================

# ---------------------------------------------------------------------------
# 工程设置
# ---------------------------------------------------------------------------
set project_name  "smolvlm2_decoder"
set solution_name "sol_200mhz"
set part          "xck26-sfvc784-2LV-c"
set clock_period  "5ns"          ;# 200 MHz
set tb_file       "../tb/decoder_tb.cpp"

# 清理旧工程
if {[file exists $project_name]} {
    file delete -force $project_name
}

open_project $project_name
set_top smolvlm2_decoder_layer

# ---------------------------------------------------------------------------
# 源文件（头文件 include 在 .cpp 内，只需添加 .cpp）
# ---------------------------------------------------------------------------
add_files ../src/decoder_kernel.cpp \
    -cflags "-I../src -O2 -std=c++14"

# Testbench
add_files -tb $tb_file \
    -cflags "-I../src -O2 -std=c++14"

# ---------------------------------------------------------------------------
# 解决方案：目标 part + 时钟
# ---------------------------------------------------------------------------
open_solution $solution_name -flow_target vitis
set_part $part
create_clock -period $clock_period -name default

# ---------------------------------------------------------------------------
# HLS 方向性优化配置
# ---------------------------------------------------------------------------

# 强制 256-bit AXI 接口（防止 HLS 退化为窄口）
config_interface -m_axi_addr64

# 流水线策略：激进展开（DATAFLOW 下各子函数自主 PIPELINE）
config_compile -pipeline_style flp

# DATAFLOW：检查通道宽度匹配
config_dataflow -strict_mode error

# array to stream 转换（hls::stream 后备）
config_array_partition -auto_partition_threshold 128

# ---------------------------------------------------------------------------
# C 仿真（csim）
# ---------------------------------------------------------------------------
csim_design -clean -O

# ---------------------------------------------------------------------------
# C 综合（csynth）
# ---------------------------------------------------------------------------
csynth_design

# ---------------------------------------------------------------------------
# 综合报告检查（辅助诊断）
# ---------------------------------------------------------------------------
# 读取综合报告（Tcl 方式）
set rpt_dir "${project_name}/${solution_name}/syn/report"
if {[file exists "${rpt_dir}/smolvlm2_decoder_layer_csynth.rpt"]} {
    set fp [open "${rpt_dir}/smolvlm2_decoder_layer_csynth.rpt" r]
    set content [read $fp]
    close $fp
    # 打印资源摘要（DSP/BRAM/URAM 行）
    foreach line [split $content "\n"] {
        if {[regexp {DSP|BRAM|URAM|LUT|FF|Timing} $line]} {
            puts "  [string trim $line]"
        }
    }
}

# ---------------------------------------------------------------------------
# RTL 仿真（co-sim，可选，时间较长）
# ---------------------------------------------------------------------------
# cosim_design -O -trace_level all

# ---------------------------------------------------------------------------
# 导出 IP（XO 文件，供 v++ 链接）
# ---------------------------------------------------------------------------
export_design -flow syn -rtl verilog \
    -format xo \
    -output "${project_name}_kernel.xo"

puts "============================================================"
puts "  HLS synthesis complete."
puts "  XO file: ${project_name}_kernel.xo"
puts "  Check DSP > 600, URAM < 51, BRAM < 115, WNS > 0"
puts "============================================================"

close_project
