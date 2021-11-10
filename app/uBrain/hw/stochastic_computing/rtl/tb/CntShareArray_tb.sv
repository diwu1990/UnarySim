`timescale 1ns/1ns
`include "../CntShareArray.sv"

module CntShareArray_tb ();
    parameter IDIM = 2;
    parameter CWID = 8;
    parameter SDIM = 8;

    logic   clk;
    logic   rst_n;
    logic   enable [IDIM - 1 : 0];
    logic   [CWID - 1 : 0] cntSeq [IDIM * SDIM - 1 : 0];

    CntShareArray #(
        .IDIM(IDIM),
        .CWID(CWID),
        .SDIM(SDIM)
    ) U_CntShareArray(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .cntSeq(cntSeq)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("CntShareArray.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst_n = 0;
        enable = {1'b1, 1'b1};
        
        #15;
        rst_n = 1;

        #400;
        enable = {1'b1, 1'b0};

        #400;
        $finish;
    end

endmodule