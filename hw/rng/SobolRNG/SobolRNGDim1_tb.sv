`timescale 1ns/1ns
`include "SobolRNGDim1.sv"
module SobolRNGDim1_tb ();

    logic   clk;
    logic   rst_n;
    logic   enable;
    logic   [`INWD-1:0]sobolSeq;

    SobolRNGDim1 U_SobolRNGDim1(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .sobolSeq(sobolSeq)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("SobolRNGDim1.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst_n = 0;
        enable = 1;
        
        #15;
        rst_n = 1;
        #400;
        $finish;
    end

endmodule