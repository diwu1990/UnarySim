`include "SobolRngDim1.sv"

module RngShareArray #(
    parameter IDIM = 16, // rng dimension, i.e., rng count
    parameter RWID = 8, // rng width
    parameter SDIM = 32 // sharing dimension
) (
    input logic clk,
    input logic rst_n,
    input logic enable [IDIM - 1 : 0],
    output logic [RWID - 1 : 0] rngSeq [IDIM * SDIM - 1 : 0]
);

    logic [RWID - 1 : 0] sobolSeq [IDIM - 1 : 0];

    genvar i, j;
    generate
        for (i = 0; i < IDIM; i = i + 1) begin
            SobolRngDim1 #(
                .RWID(RWID)
            ) U_SobolRngDim1(
                .clk(clk),
                .rst_n(rst_n),
                .enable(enable[i]),
                .sobolSeq(sobolSeq[i])
            );
            for (j = 0; j < SDIM; j = j + 1) begin
                assign rngSeq[i * SDIM + j] = sobolSeq[i];
            end
        end
    endgenerate
    
endmodule