`timescale 1ns/1ns
`include "dMUL_uni.sv"

module dMUL_uni_tb ();

    logic   clk;
    logic   rst_n;
    logic   [`INWD-1:0] iA;
    logic   [`INWD-1:0] iB;
    logic   loadA;
    logic   loadB;
    logic   oC;

    dMUL_uni U_dMUL_uni(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .iA(iA),
        .iB(iB),
        .loadA(loadA),
        .loadB(loadB),
        .oC(oC)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("dMUL_uni.fsdb");
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
        loadA = 0;
        loadB = 0;

        #15;
        rst_n = 1;

        #10;
        loadA = 1;
        loadB = 1;

        #10;
        loadA = 0;
        loadB = 0;

        #1000;
        $finish;
    end

endmodule