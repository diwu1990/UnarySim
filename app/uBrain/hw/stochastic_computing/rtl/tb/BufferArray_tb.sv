`timescale 1ns/1ns
`include "../BufferArray.sv"

module BufferArray_tb ();

    logic   clk;
    logic   rst_n;
    logic   sel;
    logic   clear;
    logic   in [3:0];
    logic   [31:0] out [3:0];

    BufferArray # (
        .IDIM(4),
        .IWID(1),
        .OWID(32)
    ) U_BufferArray(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .iAccSel(sel),
        .iClear(clear),
        .iData(in),
        .oData(out)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("BufferArray.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif

    initial
    begin
        clk = 1;
        rst_n = 0;
        sel = 0;
        clear = 0;
        in = {1'b0, 1'b0, 1'b0, 1'b0};

        #100;
        rst_n = 1;

        #100;
        sel = 0;
        in = {1'b1, 1'b1, 1'b1, 1'b1};
        $strobe(out[0], out[1], out[2], out[3]);

        #200;
        sel = 1;
        in = {1'b1, 1'b1, 1'b1, 1'b1};
        $strobe(out[0], out[1], out[2], out[3]);

        #100;
        sel = 0;
        clear = 1;

        #10;
        clear = 0;
        in = {1'b1, 1'b1, 1'b1, 1'b1};
        $strobe(out[0], out[1], out[2], out[3]);

        #200;
        sel = 1;
        clear = 1;

        #10;
        clear = 0;
        in = {1'b1, 1'b1, 1'b1, 1'b1};
        $strobe(out[0], out[1], out[2], out[3]);
        
        #1000;
        $finish;
    end

endmodule