`include "CntEn.sv"

module CntShareArray #(
    parameter CWID = 8, // cnt width
    parameter BDIM = 16, // buffer dimension
    parameter SDIM = 32 // sharing dimension for each buffer
) (
    input logic clk,
    input logic rst_n,
    input logic enable,
    output logic [CWID - 1 : 0] cntSeq [BDIM * SDIM - 1 : 0]
);

    logic [CWID - 1 : 0] cnt;
    logic [CWID - 1 : 0] cntBuf [BDIM - 1 : 0];

    CntEn #(
        .CWID(CWID)
    ) U_CntEn(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .cnt(cnt)
    );

    genvar i, j;
    generate
        for (i = 0; i < BDIM; i = i + 1) begin
            always @(posedge clk or negedge rst_n) begin
                if (~rst_n) begin
                    cntBuf[i] <= 'b0;
                end else begin
                    cntBuf[i] <= cnt;
                end
            end
            for (j = 0; j < SDIM; j = j + 1) begin
                assign cntSeq[i * SDIM + j] = cntBuf[i];
            end
        end
    endgenerate
    
endmodule