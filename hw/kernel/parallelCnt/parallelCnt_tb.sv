`timescale 1ns/1ns
`include "parallelCnt.sv"

module parallelCnt_tb ();

    logic   clk;
    logic   rst_n;
    logic   [6:0] in;
    logic   [2:0] out;

    parallelCnt7 U_parallelCnt(
        .in(in),
        .out(out)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("parallelCnt.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst_n = 0;
        in = 1;

        #15;
        rst_n = 1;

        #20;
        in = 2;

        #10;
        in = 3;

        #10;
        in = 127;

        #100;
        in = 65;
        
        #100;
        $finish;
    end

endmodule