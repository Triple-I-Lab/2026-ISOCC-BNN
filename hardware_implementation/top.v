`include "neuron_pipelined.v"

`timescale 1ns / 1ps
module top (
    input            clk,                 // clock required for BRAM and pipelined neurons
    input            start,               // begin loading a new image
    input      [7:0] pixel_in,            // 8-bit input 784/98 cycles reduces IO from 784 to 8
    output reg [3:0] classification = 0,  // predicted MNIST digit 0-9
    output reg       done = 0             // pulses high for classification
);
  // FSM states
  // WAIT1 capture layer 1 results
  // WAIT2 layer 2 stage 1 fills from the new stable input
  // WAIT3 layer 2 stage 2 fills, l2_outs become stable next cycle
  // three WAIT states are needed because neuron_pipelined has 2 internal stages
  localparam IDLE = 0, LOAD = 1, CALC = 2, WAIT1 = 3, WAIT2 = 4, WAIT3 = 5, FINISH = 6;
  reg [2:0] state = 0;  // 3 bits to hold 7 states

  // 784 bits loaded 8 bits per cycle over 98 cycles
  reg [783:0] image_buf = 0;
  reg [6:0] load_cnt = 0;  // 7 bits to count 0 to 97

  // Layer 1 weight memory
  (* ram_style = "block" *)
  reg [783:0] w1_mem[0:255];  // forced to BRAM to avoid using thousands of LUTs
  initial $readmemb("../models/weights_layer1.txt", w1_mem);

  // Layer 2 weight memory (registers only 10 entries, small enough to skip BRAM)
  reg [255:0] w2_mem[0:9];
  initial $readmemb("../models/weights_layer2.txt", w2_mem);

  // counts up to 258. 256 neurons + 2 extra cycles to drain the 2-stage pipeline
  reg [  8:0] pipeline_cnt = 0;

  // Gated BRAM read: only active during CALC to reduce switching power
  reg [783:0] w1_dout_reg = 0;
  always @(posedge clk) begin
    if (state == CALC)
      w1_dout_reg <= w1_mem[pipeline_cnt];  // register the output for a clean 1-cycle BRAM latency
  end

  // wires for clock gating
  wire l1_ce = (state == CALC);  // stage 1 only
  wire l2_ce = (state == WAIT2) || (state == WAIT3);  // stage 2 only

  // Layer 1
  wire signed [31:0] l1_count;
  neuron_pipelined #(
      .IN(784)
  ) l1_inst (
      .clk   (clk),
      .ce    (l1_ce),
      .weight(w1_dout_reg),  // weight read from BRAM, registered
      .x     (image_buf),    // full 784-bit image, stable throughout CALC
      .count (l1_count)      // signed popcount result after 2 pipeline stages
  );

  // Layer 1 results
  reg [255:0] l1_results = 0;
  // l1_results_stable only updates ONCE after CALC completes so layer 2 neurons
  // do not recompute 256 times on a partially-filled l1_results during CALC
  reg [255:0] l1_results_stable = 0;

  // Layer 2: 10 pipelined neurons, all operating on the stable snapshot
  // outputs of neuron_pipelined are registered, so l2_outs are stable flip-flop outputs
  wire signed [31:0] l2_outs[0:9];
  genvar k;
  generate
    for (k = 0; k < 10; k = k + 1) begin : l2_gen
      neuron_pipelined #(
          .IN(256)
      ) l2_inst (
          .clk   (clk),
          .ce    (l2_ce),
          .weight(w2_mem[k]),          // layer 2 weight for digit k
          .x     (l1_results_stable),  // stable layer 1 activation output
          .count (l2_outs[k])          // signed score for digit k
      );
    end
  endgenerate

  // Parallel argmax tree
  // replaces the serial for-loop which created a ~16ns comparison chain
  // tournament-style bracket: each level halves the number of candidates
  // critical path is now ~6ns instead of ~16ns
  // level 1: 10 candidates (5 comparisons)
  // level 2:  5 winners (2 comparisons + 1 passthrough)
  // level 3:  3 winners (1 comparison + 1 passthrough)
  // level 4:  2 winners (1 final comparison)

  // level 1
  wire signed [31:0] v1_0, v1_1, v1_2, v1_3, v1_4;
  wire [3:0] i1_0, i1_1, i1_2, i1_3, i1_4;

  assign v1_0 = (l2_outs[0] >= l2_outs[1]) ? l2_outs[0] : l2_outs[1];
  assign i1_0 = (l2_outs[0] >= l2_outs[1]) ? 4'd0 : 4'd1;

  assign v1_1 = (l2_outs[2] >= l2_outs[3]) ? l2_outs[2] : l2_outs[3];
  assign i1_1 = (l2_outs[2] >= l2_outs[3]) ? 4'd2 : 4'd3;

  assign v1_2 = (l2_outs[4] >= l2_outs[5]) ? l2_outs[4] : l2_outs[5];
  assign i1_2 = (l2_outs[4] >= l2_outs[5]) ? 4'd4 : 4'd5;

  assign v1_3 = (l2_outs[6] >= l2_outs[7]) ? l2_outs[6] : l2_outs[7];
  assign i1_3 = (l2_outs[6] >= l2_outs[7]) ? 4'd6 : 4'd7;

  assign v1_4 = (l2_outs[8] >= l2_outs[9]) ? l2_outs[8] : l2_outs[9];
  assign i1_4 = (l2_outs[8] >= l2_outs[9]) ? 4'd8 : 4'd9;

  // level 2
  wire signed [31:0] v2_0, v2_1, v2_2;
  wire [3:0] i2_0, i2_1, i2_2;

  assign v2_0 = (v1_0 >= v1_1) ? v1_0 : v1_1;
  assign i2_0 = (v1_0 >= v1_1) ? i1_0 : i1_1;

  assign v2_1 = (v1_2 >= v1_3) ? v1_2 : v1_3;
  assign i2_1 = (v1_2 >= v1_3) ? i1_2 : i1_3;

  assign v2_2 = v1_4;  // passes through unchanged
  assign i2_2 = i1_4;

  // level 3
  wire signed [31:0] v3_0, v3_1;
  wire [3:0] i3_0, i3_1;

  assign v3_0 = (v2_0 >= v2_1) ? v2_0 : v2_1;
  assign i3_0 = (v2_0 >= v2_1) ? i2_0 : i2_1;

  assign v3_1 = v2_2;  // passes through
  assign i3_1 = i2_2;

  // level 4
  wire [3:0] argmax_result;
  assign argmax_result = (v3_0 >= v3_1) ? i3_0 : i3_1;

  // State machine
  always @(posedge clk) begin
    case (state)

      IDLE: begin
        done         <= 0;
        pipeline_cnt <= 0;
        if (start) begin
          image_buf <= {
            image_buf[775:0], pixel_in
          };  // 784 - 8 - 1 = 775, shift left and insert new byte at LSB
          load_cnt <= 1;  // byte 0 is captured here so LOAD starts counting from 1
          state <= LOAD;
        end
      end

      LOAD: begin
        image_buf <= {image_buf[775:0], pixel_in};  // shift in one byte per cycle, MSB first
        load_cnt  <= load_cnt + 1;
        if (load_cnt == 97) state <= CALC;
      end

      // pipeline timing with 3-cycle total offset
      //   cnt=0 address weight[0] in BRAM
      //   cnt=1 w1_dout_reg = weight[0], stage 1 of neuron samples inputs
      //   cnt=2 partial[] for weight[0] ready, stage 2 of neuron samples
      //   cnt=3 l1_count = result for weight[0], save to l1_results[255]
      //   cnt=258 result for weight[255] saved to l1_results[0], done with layer 1
      CALC: begin
        pipeline_cnt <= pipeline_cnt + 1;
        if (pipeline_cnt >= 3 && pipeline_cnt <= 258)
          l1_results[255 - (pipeline_cnt - 3)] <= (l1_count >= 0); // positive maps to 1, negative maps to 0
        if (pipeline_cnt == 258) state <= WAIT1;
      end

      // capture completed layer 1 results into stable register
      // layer 2 pipeline will now begin filling from this stable value
      WAIT1: begin
        l1_results_stable <= l1_results;
        state <= WAIT2;
      end

      WAIT2: state <= WAIT3;  // layer 2 stage 1 processes new l1_results_stable this cycle

      WAIT3: state <= FINISH;  // layer 2 stage 2 processes stage 1 output

      // l2_outs are now stable registered values
      // argmax_result is fully resolved by the combinational tree before this posedge
      FINISH: begin
        classification <= argmax_result;  // register the winning digit index
        done <= 1;
        state <= IDLE;
      end

    endcase
  end
endmodule

