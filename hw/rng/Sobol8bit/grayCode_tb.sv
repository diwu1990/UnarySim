`timescale 1ns/1ns
`include "grayCode.sv"
module grayCode_tb ();

    logic   clk;
    logic   rst;
    logic   [`GCWD-1:0]grayOutBin;

	grayCode U_grayCode(
        .clk(clk),
        .rst(rst),
        .grayOutBin(grayOutBin)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("grayCode.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst = 1;
        
        #15;
        rst = 0;
        #200;
        $finish;
    end

endmodule