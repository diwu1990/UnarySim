`include "Cmp.sv"

module Mul #(
    parameter IWID = 8
) (
    input logic iDbit,
    input logic iDb_n,
    input logic [IWID - 1 : 0] iWeig,
    input logic [IWID - 1 : 0] iWRng,
    input logic [IWID - 1 : 0] iWR_n,
    input logic oDbit
);

    logic wBit;
    logic wB_n;

    Cmp #(
        .IWID(IWID)
    ) U_Cmp(
        .iData(iWeig),
        .iRng(iWRng),
        .oBit(wBit)
    );

    Cmp #(
        .IWID(IWID)
    ) U_Cmp_n(
        .iData(iWeig),
        .iRng(iWR_n),
        .oBit(wB_n)
    );
    
    assign oDbit = (iDbit & wBit) | (iDb_n & ~wB_n);

endmodule