`timescale 1ns/1ns
`include "uSADD16.sv"

module uSADD16_tb ();

    logic   clk;
    logic   rst_n;
    logic   [15:0] in;
    logic   out;

    uSADD16 U_uSADD16(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(in),
        .out(out)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("uSADD16.fsdb");
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
        in = 16'b1111000000000000;
        
        #100;
        $finish;
    end

endmodule