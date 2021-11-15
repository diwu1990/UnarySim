`include "Dff.sv"

module Inverter #(
) (
    input logic clk,
    input logic rst_n,
    input logic iBit,
    output logic oBit
);

    logic tBit;

    assign tBit = ~iBit;

    Dff U_Dff(
        .clk(clk),
        .rst_n(rst_n),
        .in(tBit),
        .out(oBit)
    );

endmodule