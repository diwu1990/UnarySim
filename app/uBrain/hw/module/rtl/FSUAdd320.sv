`include "AdderTree.sv"

module FSUAdd320 #(
    parameter IDIM = 320,
    parameter IWID = 1,
    parameter SCAL = 1,
    parameter TSCL = SCAL * 2,
    parameter OFST = IDIM - 1,
    parameter BDEP = 2
) (
    input logic clk,
    input logic rst_n,
    input logic iBit [IDIM - 1 : 0],
    output logic oBit
);

    logic [$clog2(IDIM) + 1 - 1: 0] pc;
    logic [$clog2(IDIM) + 1 + IWID - 1 + 1 : 0] acc;
    logic [$clog2(IDIM) + 1 + IWID - 1 + 1 : 0] acc_new;

    AdderTree #(
        .IDIM(IDIM),
        .IWID(1),
        .BDEP(BDEP)
    ) U_AdderTree_parallel_counter(
        .clk(clk),
        .rst_n(rst_n),
        .iData(iBit),
        .oData(pc)
    );

    assign acc_new = acc + {pc, 1'b0} - OFST;
    assign oBit = (acc_new >= TSCL);
    assign acc_mod = oBit ? (acc_new - SCAL) : acc_new;

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            acc <= 0;
        end
        else begin
            acc <= acc_mod;
        end
    end

endmodule