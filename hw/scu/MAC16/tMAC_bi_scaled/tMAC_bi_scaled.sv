`include "SobolRNGDim1_4b.sv"
`include "tMUL_bi.sv"
`include "muxADD.sv"

module tMAC_bi_scaled (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input logic [7:0] iA [15:0],
    input logic [7:0] iB [15:0],
    input logic loadA,
    input logic loadB,
    output oC
);
    // logic [7:0] sobolSeqA;
    // logic [7:0] sobolSeqB;
    logic [15:0] mulC;
    
    // SobolRNGDim1_8b U_SobolRNGDim1_8b_A(
    //     .clk(clk),
    //     .rst_n(rst_n),
    //     .enable(1'b1),
    //     .sobolSeq(sobolSeqA)
    //     );

    // SobolRNGDim1_8b U_SobolRNGDim1_8b_B(
    //     .clk(clk),
    //     .rst_n(rst_n),
    //     .enable(&cntB),
    //     .sobolSeq(sobolSeqB)
    //     );

    genvar i;
    generate
        for (i = 0; i < 16; i++) begin
            tMUL_bi U_tMUL_bi(
                .clk(clk),    // Clock
                .rst_n(rst_n),  // Asynchronous reset active low
                .iA(iA[i]),
                .iB(iB[i]),
                .loadA(loadA),
                .loadB(loadB),
                .oC(mulC[i]),
                .stop(stop)
                );
        end
    endgenerate

    logic [3:0] sel;
    SobolRNGDim1_4b U_SobolRNGDim1_4b(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .sobolSeq(sel)
        );

    muxADD U_muxADD(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(mulC),
        .sel(sel),
        .out(oC)
        );

endmodule