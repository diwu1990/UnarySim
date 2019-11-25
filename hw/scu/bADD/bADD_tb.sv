`timescale 1ns/1ns
`include "bADD.sv"

module bADD_tb ();

    logic   clk;
    logic   rst_n;
    logic   [`DATAWD-1:0] iA;
    logic   [`DATAWD-1:0] iB;
    // logic   loadA;
    // logic   loadB;
    logic [`DATAWD:0] oC;
    // logic   stop;

    bADD U_bADD(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .iA(iA),
        .iB(iB),
        .oC(oC)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("bADD.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst_n = 0;
        iA = 128;
        iB = 128;

        #15;
        rst_n = 1;

        #1000;
        $finish;
    end

endmodule