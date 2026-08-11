`timescale 1ns / 1ps
module neuron_pipelined #(
    parameter IN     = 784,  // input width: 784 for layer 1, 256 for layer 2
    parameter GROUPS = 16    // number of parallel groups to split the popcount into
) (
    input                      clk,
    input                      ce,      // clock enable: gates both pipeline stages
    input             [IN-1:0] weight,  // binary weight vector
    input             [IN-1:0] x,       // binary input vector
    output reg signed [  31:0] count    // signed output: range is [-IN, +IN]
);
  // GROUP_SIZE = ceil(IN / GROUPS)
  // for IN=784, GROUPS=16: GROUP_SIZE = 49  (exact: 16 * 49 = 784)
  // for IN=256, GROUPS=16: GROUP_SIZE = 16  (exact: 16 * 16 = 256)
  localparam GROUP_SIZE = (IN + GROUPS - 1) / GROUPS;
  localparam PBITS = 7;  // 7 bits holds 0 to 127, safe for any GROUP_SIZE up to 127

  // XNOR binary weight multiplication
  wire [IN-1:0] mult;
  assign mult = ~(weight ^ x);  // XNOR

  // Stage 1, 16 independent partial popcounts
  // splitting into 16 groups turns 784-deep serial chain into
  // 16 parallel 49-input trees (6ns instead of 26ns which was old critical)
  // Fmax improvement needed from 38 to 82 MHz for ARTIX-7, results differ on UltraScale+
  reg [PBITS-1:0] partial[0:GROUPS-1];
  integer g, b;
  reg [PBITS-1:0] p_temp;  // local accumulator per group

  always @(posedge clk) begin
    if (ce) begin
      for (g = 0; g < GROUPS; g = g + 1) begin
        p_temp = 0;
        for (b = 0; b < GROUP_SIZE; b = b + 1) begin
          if ((g * GROUP_SIZE + b) < IN)  // out of range on last group
            p_temp = p_temp + mult[g*GROUP_SIZE+b];
        end
        partial[g] <= p_temp;
      end
    end
  end

  // Stage 2, sum the 16 partial results, compute final count
  // max possible sum = 16 groups * 49 bits = 784, fits in 32-bit s_temp
  integer i;
  reg [31:0] s_temp;

  always @(posedge clk) begin
    if (ce) begin
      s_temp = 0;

      for (i = 0; i < GROUPS; i = i + 1)
      s_temp = s_temp + partial[i];  // accumulate all partial popcounts

      // extends popcount by 1 bit to avoid overflow, left shifts by 1,
      // then subtracts IN, this maps popcount to signed BNN score in range [-IN, +IN]
      count <= $signed({1'b0, s_temp} << 1) - $signed(IN);
    end
  end
endmodule
