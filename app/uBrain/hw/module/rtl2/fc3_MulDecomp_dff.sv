`include "fc3_MulDecomp.sv"
`include "dff.sv"

module fc3_MulDecomp_dff #(
    parameter IDIM = 1,
    parameter FOLD = 1,
    parameter ODIM = 1
) (
    input logic clk,
    input logic rstn,
    input logic iBit [IDIM - 1 : 0],
    input logic wBit [ODIM / FOLD * IDIM - 1 : 0],
    output logic oFmbs [ODIM / FOLD * IDIM - 1 : 0],
    output logic enable [ODIM / FOLD * IDIM - 1 : 0]
);
    logic oFmbs_ [ODIM / FOLD * IDIM - 1 : 0];
    logic enable_ [ODIM / FOLD * IDIM - 1 : 0];
    
    fc3_MulDecomp # (
        .IDIM(IDIM),
        .FOLD(FOLD),
        .ODIM(ODIM)
    ) U_Mul(
        .iBit(iBit),
        .wBit(wBit),
        .oFmbs(oFmbs_),
        .enable(enable_)
    );
    dff # (
        .BW(1),
        .WIDTH(1),
        .HEIGHT(1)
    ) U_dff_out(
        .clk(clk),
        .rstn(rstn),
        .d(oFmbs_),
        .q(oFmbs)
    );

    dff # (
        .BW(1),
        .WIDTH(1),
        .HEIGHT(1)
    ) U_dff_en(
        .clk(clk),
        .rstn(rstn),
        .d(enable_),
        .q(enable)
    );

endmodule
    