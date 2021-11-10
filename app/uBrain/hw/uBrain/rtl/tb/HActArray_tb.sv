`timescale 1ns/1ns
`include "../HActArray.sv"

module HActArray_tb ();

    logic   clk;
    logic   rst_n;
    logic   [11:0] in [0:0];
    logic   [7:0] out [0:0];

    HActArray # (
        .IDIM(1),
        .IWID(12),
        .ADIM(16),
        .OWID(8),
        .RELU(1)
    ) U_HActArray(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .iData(in),
        .oData(out)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("HActArray.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif

    initial
    begin
        clk = 1;
        rst_n = 0;
        in = {12'd0};

        #100;
        rst_n = 1;

        #100;
        in = {12'd255};
        $strobe(out[0]);

        #100;
        in = {12'd256};
        $strobe(out[0]);

        #100;
        in = {12'd257};
        $strobe(out[0]);

        #100;
        in = {12'd511};
        $strobe(out[0]);

         #100;
        in = {12'd512};
        $strobe(out[0]);

         #100;
        in = {12'd513};
        $strobe(out[0]);

        #100;
        in = {12'd1023};
        $strobe(out[0]);

        #100;
        in = {12'd1024};
        $strobe(out[0]);

        #100;
        in = {12'd1025};
        $strobe(out[0]);

        #100;
        in = {12'd2048 - 12'd129};
        $strobe(out[0]);

        #100;
        in = {12'd2048 - 12'd128};
        $strobe(out[0]);

        #100;
        in = {12'd2048 - 12'd127};
        $strobe(out[0]);

        #100;
        in = {12'd2047};
        $strobe(out[0]);

        #100;
        in = {12'd2048};
        $strobe(out[0]);

        #100;
        in = {12'd2049};
        $strobe(out[0]);

        #100;
        in = {12'd2048 + 12'd127};
        $strobe(out[0]);

        #100;
        in = {12'd2048 + 12'd128};
        $strobe(out[0]);

        #100;
        in = {12'd2048 + 12'd129};
        $strobe(out[0]);

        #100;
        in = {12'd4095};
        $strobe(out[0]);
        
        #1000;
        $finish;
    end

endmodule