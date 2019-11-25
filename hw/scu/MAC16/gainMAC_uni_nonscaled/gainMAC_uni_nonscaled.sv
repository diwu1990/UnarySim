`include "SobolRNGDim1_8b.sv"
`include "orADD.sv"
`include "gMUL_uni.sv"

module gainMAC_uni_nonscaled (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input logic [7:0] iA [15:0],
    input logic [7:0] iB [15:0],
    input logic loadA,
    input logic loadB,
    output oC
);
    logic [7:0] sobolSeqA [15:0];
    logic [7:0] sobolSeqB [15:0];
    logic [15:0] mulC;

    genvar i;
    generate
        for (i = 0; i < 16; i++) begin
            SobolRNGDim1_8b U_SobolRNGDim1_8b_A(
                .clk(clk),
                .rst_n(rst_n),
                .enable(1'b1),
                .sobolSeq(sobolSeqA[i])
                );

            SobolRNGDim1_8b U_SobolRNGDim1_8b_B(
                .clk(clk),
                .rst_n(rst_n),
                .enable(1'b1),
                .sobolSeq(sobolSeqB[i])
                );

            gMUL_uni U_gMUL_uni(
                .clk(clk),    // Clock
                .rst_n(rst_n),  // Asynchronous reset active low
                .iA(iA[i]),
                .iB(iB[i]),
                .loadA(loadA),
                .loadB(loadB),
                .sobolSeqA(sobolSeqA[i]),
                .sobolSeqB(sobolSeqB[i]),
                .oC(mulC[i])
                );
        end
    endgenerate

    orADD U_orADD(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(mulC),
        .out(oC)
        );

endmodule