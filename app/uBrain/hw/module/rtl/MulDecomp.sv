`include "Dff.sv"

module MulDecomp #(
) (
    input logic clk,
    input logic rst_n,
    input logic iBit0,
    input logic iBit1,
    output logic oInv0,
    output logic oBit
);

    logic tBit;
    logic tInv0;

    assign tInv0 = ~iBit0;
    assign tBit = (iBit0 & iBit1) | (tInv0 & ~iBit1);

    Dff U_Dff_oBit(
        .clk(clk),
        .rst_n(rst_n),
        .in(tBit),
        .out(oBit)
    );

    Dff U_Dff_oInv0(
        .clk(clk),
        .rst_n(rst_n),
        .in(tInv0),
        .out(oInv0)
    );

endmodule