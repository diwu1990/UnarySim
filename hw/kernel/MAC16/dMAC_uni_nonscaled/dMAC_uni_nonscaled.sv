`include "SobolRNGDim1_8b.sv"
`include "orADD.sv"
`include "dMUL_uni.sv"

module dMAC_uni_nonscaled (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input logic [7:0] iA [15:0],
    input logic [7:0] iB [15:0],
    input logic loadA,
    input logic loadB,
    output oC
);
    logic [7:0] sobolSeqA;
    logic [7:0] sobolSeqB;
    logic [7:0] cntA;
    logic [7:0] cntB;
    logic [15:0] mulC;
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_cntA
        if(~rst_n) begin
            cntA <= 0;
        end else begin
            cntA <= cntA + 1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_cntB
        if(~rst_n) begin
            cntB <= 0;
        end else begin
            cntB <= cntB + (&cntA);
        end
    end

    SobolRNGDim1_8b U_SobolRNGDim1_8b_A(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .sobolSeq(sobolSeqA)
        );

    SobolRNGDim1_8b U_SobolRNGDim1_8b_B(
        .clk(clk),
        .rst_n(rst_n),
        .enable(&cntB),
        .sobolSeq(sobolSeqB)
        );

    genvar i;
    generate
        for (i = 0; i < 16; i++) begin
            dMUL_uni U_dMUL_uni(
                .clk(clk),    // Clock
                .rst_n(rst_n),  // Asynchronous reset active low
                .iA(iA[i]),
                .iB(iB[i]),
                .loadA(loadA),
                .loadB(loadB),
                .sobolSeqA(sobolSeqA),
                .sobolSeqB(sobolSeqB),
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