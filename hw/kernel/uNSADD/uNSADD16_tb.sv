`timescale 1ns/1ns
`include "uNSADD16.sv"

module uNSADD16_tb ();

    logic   clk;
    logic   rst_n;
    logic   [15:0] in;
    logic   out;

    uNSADD16 U_uNSADD16(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(in),
        .out(out)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("uNSADD16.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst_n = 0;
        in = 16'b1111111111111111;

        #15;
        rst_n = 1;

        #20;
        in = 16'b1111111100000000;

        #100;
        in = 16'b1111000000000000;

        #100;
        in = 16'b0000000000000000;
        
        #1000;
        $finish;
    end

endmodule