`include "CntEn.sv"

module CntShareArray #(
    parameter IDIM = 16, // cnt dimension, i.e., cnt count
    parameter CWID = 8, // cnt width
    parameter SDIM = 32 // sharing dimension
) (
    input logic clk,
    input logic rst_n,
    input logic enable [IDIM - 1 : 0],
    output logic [CWID - 1 : 0] cntSeq [IDIM * SDIM - 1 : 0]
);

    logic [CWID - 1 : 0] cnt [IDIM - 1 : 0];

    genvar i, j;
    generate
        for (i = 0; i < IDIM; i = i + 1) begin
            CntEn #(
                .CWID(CWID)
            ) U_CntEn(
                .clk(clk),
                .rst_n(rst_n),
                .enable(enable[i]),
                .cnt(cnt[i])
            );
            for (j = 0; j < SDIM; j = j + 1) begin
                assign cntSeq[i * SDIM + j] = cnt[i];
            end
        end
    endgenerate
    
endmodule