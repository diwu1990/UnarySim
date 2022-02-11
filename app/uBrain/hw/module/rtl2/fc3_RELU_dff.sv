`include "fc3_RELU.sv"
`include "dff.sv"

module fc3_RELU_dff #(
    parameter IDIM = 1, 
    parameter IWID = $clog2(110*32) + 1 + 10,
    parameter ADIM = 110*32, // accumulation depth
    parameter ODIM = IDIM,
    parameter OWID = 10,
    parameter PZER = ADIM * (2**OWID) / 2,
    parameter PPON = ADIM * (2**OWID) / 2 + (2**OWID) / 2,
    parameter PNON = ADIM * (2**OWID) / 2 - (2**OWID) / 2
    //parameter RELU = 1
) (
    input logic clk,
    input logic rst_n,
    input logic [IWID - 1 : 0] iData [IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oData [ODIM - 1 : 0]
);
    logic [OWID - 1 : 0] oData_ [ODIM - 1 : 0];

    fc3_RELU # (
        .IDIM(IDIM), 
        .IWID(IWID),
        .ADIM(ADIM),
        .ODIM(ODIM),
        .OWID(OWID),
        .PZER(PZER),
        .PPON(PPON),
        .PNON(PNON)
    ) U_relu (
        .iData(iData),
        .oData(oData_)
    );
    dff # (
        .BW(OWID),
        .WIDTH(1),
        .HEIGHT(ODIM)
    ) U_dff(
        .clk(clk),
        .rstn(rstn),
        .d(oData_),
        .q(oData)
    );

endmodule