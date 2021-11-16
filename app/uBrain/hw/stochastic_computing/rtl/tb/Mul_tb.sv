`timescale 1ns/1ns
`include "../Mul.sv"

module Mul_tb ();

    parameter IWID = 4;

    logic iDbit;
    logic iDb_n;
    logic [IWID - 1 : 0] iWeig;
    logic [IWID - 1 : 0] iWRng;
    logic [IWID - 1 : 0] iWR_n;
    logic oDbit;

    Mul # (
        .IWID(IWID)
    ) U_Mul(
        .iDbit(iDbit),
        .iDb_n(iDb_n),
        .iWeig(iWeig),
        .iWRng(iWRng),
        .iWR_n(iWR_n),
        .oDbit(oDbit)
        );

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("Mul.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif

    always # 10 iWRng = iWRng + 1;

    always # 10 iWR_n = iWR_n + 1;

    initial
    begin
        iDbit = 'd1;
        iDb_n = 'd0;
        iWeig = 'd10;

        iWRng = 'd7;
        iWR_n = 'd15;
        
        #200
        $finish;
    end

endmodule