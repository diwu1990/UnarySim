`include "fc3_MulXNOR.sv"
`include "dff.sv"

module fc3_MulXNOR_dff #(
    parameter IDIM = 1,
    parameter FOLD = 1,
    parameter ODIM = 1
) (
    input logic clk,
    input logic rstn,
    input logic iBit [IDIM - 1 : 0],
    input logic wBit [ODIM / FOLD * IDIM - 1 : 0],
    output logic oFmbs [ODIM / FOLD * IDIM - 1 : 0]
);
    logic oFmbs_ [ODIM / FOLD * IDIM - 1 : 0];
    
    fc3_MulXNOR # (
        .IDIM(IDIM),
        .FOLD(FOLD),
        .ODIM(ODIM)
    ) U_Mul(
        .iBit(iBit),
        .wBit(wBit),
        .oFmbs(oFmbs_)
    );
    dff # (
        .BW(1),
        .WIDTH(1),
        .HEIGHT(1)
    ) U_dff(
        .clk(clk),
        .rstn(rstn),
        .d(oFmbs_),
        .q(oFmbs)
    );

endmodule
    