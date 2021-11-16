`include "FSUAdd.sv"
`include "FSULinear.sv"
`include "HUBLinearFold.sv"

module HUBMGU #(
    parameter IDIM = 256,
    parameter IWID = 10,
    parameter ODIM = 64,
    parameter OWID = IWID,
    parameter RWID = IWID,
    parameter SDIM = 32,
    parameter BDEP = 999,
    parameter BDIM = 2 ** ($clog2(IDIM) - $clog2(SDIM)),
    parameter TDIM = (BDIM < 1) ? 1 : BDIM
) (
    input logic clk,
    input logic rst_n,
    input logic load,
    input logic sel,
    input logic clear,
    input logic [PWID - 1 : 0] part,
    input logic [IWID - 1 : 0] iFmap [IDIM - 1 : 0],
    input logic [IWID - 1 : 0] iWeig [ODIM * IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oFmap [ODIM - 1 : 0]
);

    HUBLinearFold # (
        .IDIM(IDIM),
        .IWID(IWID),
        .ODIM(ODIM),
        .RELU(RELU),
        .SDIM(SDIM),
        .FOLD(FOLD),
        .BDEP(BDEP)
    ) U_HUBLinearFold_FC3F1 (
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .load(load),
        .sel(sel),
        .clear(clear),
        .part(part),
        .iFmap(iFmap),
        .iWeig(iWeig),
        .oFmap(oFmap)
        );
    
endmodule