`timescale 1ns/1ns
`include "../CntEn.sv"

module CntEn_tb ();
    parameter CWID = 8;

    logic   clk;
    logic   rst_n;
    logic   enable;
    logic   [CWID - 1 : 0] cnt;

	CntEn U_CntEn(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .cnt(cnt)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("CntEn.fsdb");
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