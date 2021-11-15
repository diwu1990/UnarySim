`include "HAct.sv"
`include "Dff.sv"

module RELU14_10 #(
    parameter IWID = 14,
    parameter ADIM = 9,
    parameter OWID = 10
) (
    input logic clk,
    input logic rst_n,
    input logic [IWID - 1 : 0] iData,
    output logic [OWID - 1 : 0] oData
);

    logic [OWID - 1 : 0] tData;

    HAct #(
        .IWID(IWID),
        .ADIM(ADIM),
        .OWID(OWID)
    ) U_HAct(
        .iData(iData),
        .oData(tData)
    );

    Dff #(
        .IWID(OWID)
    ) U_Dff(
        .clk(clk),
        .rst_n(rst_n),
        .in(tData),
        .out(oData)
    );

endmodule