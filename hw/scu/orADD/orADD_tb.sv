`timescale 1ns/1ns
`include "orADD.sv"

module orADD_tb ();

    logic   clk;
    logic   rst_n;
    logic   [`INUM-1:0] in;
    logic   out;

    orADD U_orADD(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(in),
        .out(out)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("orADD.fsdb");
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

        #100;
        in = 3;

        #100;
        in = 0;
        
        #100;
        $finish;
    end

endmodule