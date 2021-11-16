`timescale 1ns/1ns
`include "../SobolRngDim1.sv"

module SobolRngDim1_tb ();
    parameter RWID = 8;

    logic   clk;
    logic   rst_n;
    logic   enable;
    logic   [RWID - 1 : 0]sobolSeq;

    SobolRngDim1 U_SobolRngDim1(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .sobolSeq(sobolSeq)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("SobolRngDim1.fsdb");
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