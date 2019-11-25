`timescale 1ns/1ns
`include "ModCnt5.sv"

module ModCnt5_tb ();

    logic   clk;
    logic   rst;
    logic   cin;

    ModCnt5 U_ModCnt5(
        .clk(clk),
        .rst(rst),
        .cin(cin),
        .cout(cout),
        .out(out)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("ModCnt5.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst = 1;
        cin = 0;
        
        #15;
        rst = 0;
        repeat(500) begin
            #10 cin = $random;
        end
        $finish;
    end


endmodule