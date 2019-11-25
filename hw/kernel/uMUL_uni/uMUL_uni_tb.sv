`timescale 1ns/1ns
`include "uMUL_uni.sv"

module uMUL_uni_tb ();

    logic   clk;
    logic   rst_n;
    logic   iA;
    logic   [`INWD-1:0] iB;
    logic   loadB;
    logic   oC;

    uMUL_uni U_uMUL_uni(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .iA(iA),
        .iB(iB),
        .loadB(loadB),
        .oC(oC)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("uMUL_uni.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst_n = 0;
        iB = 128;
        loadB = 0;
        iA = 0;

        #15;
        rst_n = 1;

        #10;
        loadB = 1;

        #10;
        loadB = 0;

        #50;
        iA = 1;
        #500;

        iA = 0;

        #1000;
        $finish;
    end

endmodule