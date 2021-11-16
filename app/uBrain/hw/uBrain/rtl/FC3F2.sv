`include "HUBLinearFold.sv"

module FC3F2 #(
    parameter IDIM = 110,
    parameter IWID = 10,
    parameter ODIM = 256,
    parameter OWID = IWID,
    parameter RWID = IWID,
    parameter RELU = 1,
    parameter SDIM = 32,
    parameter FOLD = 2,
    parameter PWID = ($clog2(FOLD) < 2) ? 1 : $clog2(FOLD),
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
    ) U_HUBLinearFold_FC3F2 (
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