`timescale 1ns/1ns
`include "../RngShareArray.sv"

module RngShareArray_tb ();
    parameter RWID = 8;
    parameter BDIM = 2;
    parameter SDIM = 8;

    logic   clk;
    logic   rst_n;
    logic   enable;
    logic   [RWID - 1 : 0] rngSeq [BDIM * SDIM - 1 : 0];

    RngShareArray #(
        .BDIM(BDIM),
        .RWID(RWID),
        .SDIM(SDIM)
    ) U_RngShareArray(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .rngSeq(rngSeq)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("RngShareArray.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst_n = 0;
        enable = 1'b1;
        
        #15;
        rst_n = 1;

        #400;
        enable = 1'b0;

        #400;
        $finish;
    end

endmodule