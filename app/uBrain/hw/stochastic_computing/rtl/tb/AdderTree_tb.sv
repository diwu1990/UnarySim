`timescale 1ns/1ns
`include "../AdderTree.sv"

module AdderTree_tb ();

    logic   clk;
    logic   rst_n;
    logic   in [6:0];
    logic   [3:0] out;

    AdderTree # (
        .IDIM(7),
        .IWID(1),
        .BDEP(2)
    ) U_AdderTree(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .iData(in),
        .oData(out)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("AdderTree.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif

    initial
    begin
        clk = 1;
        rst_n = 0;
        in = {1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0};
        $strobe("Expected: 0; Actual: ", out);

        #100;
        rst_n = 1;

        #100;
        in = {1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1};
        $strobe("Expected: 7; Actual: ", out);

        #100;
        in = {1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b0};
        $strobe("Expected: 6; Actual: ", out);

        #100;
        in = {1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0};
        $strobe("Expected: 3; Actual: ", out);

        #100;
        in = {1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0};
        $strobe("Expected: 0; Actual: ", out);
        
        #1000;
        $finish;
    end

endmodule