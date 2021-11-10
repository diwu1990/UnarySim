`include "SobolRngDim1.sv"

module RngShareArray #(
    parameter RWID = 8, // rng width
    parameter BDIM = 16, // buffer dimension
    parameter SDIM = 32 // sharing dimension for each buffer
) (
    input logic clk,
    input logic rst_n,
    input logic enable,
    output logic [RWID - 1 : 0] rngSeq [BDIM * SDIM - 1 : 0]
);

    logic [RWID - 1 : 0] sobolSeq;
    logic [RWID - 1 : 0] sobolBuf [BDIM - 1 : 0];

    SobolRngDim1 #(
        .RWID(RWID)
    ) U_SobolRngDim1(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .sobolSeq(sobolSeq)
    );

    genvar i, j;
    generate
        for (i = 0; i < BDIM; i = i + 1) begin
            always @(posedge clk or negedge rst_n) begin
                if (~rst_n) begin
                    sobolBuf[i] <= 'b0;
                end else begin
                    sobolBuf[i] <= sobolSeq;
                end
            end
            for (j = 0; j < SDIM; j = j + 1) begin
                assign rngSeq[i * SDIM + j] = sobolBuf[i];
            end
        end
    endgenerate
    
endmodule