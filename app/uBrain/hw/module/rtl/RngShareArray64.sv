`include "SobolRngDim1.sv"

module RngShareArray64 #(
    parameter RWID = 10, // rng width
    parameter BDIM = 2, // buffer dimension
    parameter TDIM = (BDIM < 1) ? 1 : BDIM, // true buffer dimension to deal with corner case
    parameter SDIM = 32 // sharing dimension for each buffer
) (
    input logic clk,
    input logic rst_n,
    input logic enable,
    output logic [RWID - 1 : 0] rngSeq [TDIM * SDIM - 1 : 0]
);

    logic [RWID - 1 : 0] sobolSeq;
    logic [RWID - 1 : 0] sobolBuf [TDIM - 1 : 0];

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
        for (i = 0; i < TDIM; i = i + 1) begin
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