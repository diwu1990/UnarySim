`include "Dff.sv"

module MulXNOR #(
) (
    input logic clk,
    input logic rst_n,
    input logic iBit0,
    input logic iBit1,
    output logic oBit
);

    logic tBit;

    assign tBit = ~(iBit0 ^ iBit1);

    Dff U_Dff(
        .clk(clk),
        .rst_n(rst_n),
        .in(tBit),
        .out(oBit)
    );

endmodule