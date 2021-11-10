`timescale 1ns/1ns
`include "../Cmp.sv"

module Cmp_tb ();

    parameter IWID = 4;

    logic [IWID - 1:0]iData;
    logic [IWID - 1:0]iRng;
    logic oBit;

    Cmp # (
        .IWID(IWID)
    ) U_Cmp(
        .iData(iData),
        .iRng(iRng),
        .oBit(oBit)
        );

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("Cmp.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif

    always # 10 iRng = iRng + 1;

    initial
    begin
        iData = 'd10;
        iRng = 'd0;
        
        #200
        $finish;
    end

endmodule