# constraints.xdc
#

# 82 MHz clock: 12.2 ns period
create_clock -period 12.200 -name sys_clk [get_ports clk]

# input timing: pixel_in and start must be stable 2ns before the rising clock edge
set_input_delay -clock sys_clk 2.000 [get_ports {start pixel_in[*]}]
