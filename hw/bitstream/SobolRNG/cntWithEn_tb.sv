`timescale 1ns/1ns
`include "cntWithEn.sv"
module cntWithEn_tb ();

    logic   clk;
    logic   rst_n;
    logic   enable;
    logic   [`CNTWD-1:0]cntOut;

	cntWithEn U_cntWithEn(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .cntOut(cntOut)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("cntWithEn.fsdb");
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