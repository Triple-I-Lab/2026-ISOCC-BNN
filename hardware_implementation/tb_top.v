`include "top.v"

`timescale 1ns / 1ps
module tb_top;
  reg          clk = 0;
  reg          start = 0;
  reg  [  7:0] pixel_in = 0;
  wire [  3:0] classification;
  wire         done;

  reg  [783:0] test_images    [0:9999];  // MNIST test data
  reg  [  3:0] test_labels    [0:9999];  // digit label 0-9

  integer i, b, correct;
  integer active_count, l1_idx;

  top dut (
      .clk           (clk),
      .start         (start),
      .pixel_in      (pixel_in),
      .classification(classification),
      .done          (done)
  );

  always #5 clk = ~clk;

  initial begin
    $readmemb("../models/test_data.txt", test_images);
    $readmemh("../models/test_labels.txt", test_labels);

    correct = 0;

    for (i = 0; i < 10000; i = i + 1) begin

      // send byte 0 on the same cycle as start 
      @(negedge clk);
      pixel_in = test_images[i][783:776]; // MSB first, bits [783:776] = byte 0
      start    = 1;

      // send bytes 1 through 97
      // 784 bits / 8 bits per cycle = 98 cycles total, 97 remain after byte 0
      for (b = 1; b < 98; b = b + 1) begin
        @(negedge clk);
        start    = 0;
        pixel_in = test_images[i][783 - b*8 -: 8]; // subtract byte b
      end

      // wait for design to classify before read
      wait (done === 1'b1);
      @(negedge clk);  // classification register settles

      if (classification == test_labels[i]) correct = correct + 1;

      // count how many layer 1 neurons activated
      // half (128/256) is a sign of healthy weights
      active_count = 0;
      for (l1_idx = 0; l1_idx < 256; l1_idx = l1_idx + 1) begin
        if (dut.l1_results[l1_idx] === 1'b1) active_count = active_count + 1;
      end

      $display(
          "Image: %5d/10000 | Predicted: %0d | Actual: %0d | L1 Active: %3d/256 | Accuracy: %5d / %-5d",
          i + 1, classification, test_labels[i], active_count, correct, i + 1);
    end

    $display("Final Accuracy: %0d / 100", correct);
    $finish;
  end
endmodule

