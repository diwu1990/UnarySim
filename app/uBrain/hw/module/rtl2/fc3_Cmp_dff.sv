`include "fc3_Cmp.sv"
`include "dff.sv"

module fc3_Cmp_dff #(
    parameter IWID = 10
) (
    input logic clk,
    input logic rstn,
    input logic [IWID - 1 : 0] iData,
    input logic [IWID - 1 : 0] iRng,
    output logic oBit
);
    logic oBit_;
    
    fc3_Cmp # (
        .IWID(IWID)
    ) U_Cmp(
        .iData(iData),
        .iRng(iRng),
        .oBit(oBit_)
    );
    dff # (
        .BW(1),
        .WIDTH(1),
        .HEIGHT(1)
    ) U_dff(
        .clk(clk),
        .rstn(rstn),
        .d(oBit_),
        .q(oBit)
    );

endmodule
    