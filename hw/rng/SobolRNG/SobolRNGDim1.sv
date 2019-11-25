`include "SobolRNGDef.sv"
`include "SobolRNG.sv"
`include "LSZ.sv"
`include "cntWithEn.sv"

module SobolRNGDim1 (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic enable,
    output logic [`INWD-1:0]sobolSeq
);

    logic [`INWD-1:0] cntNum;
    logic [`LOGINWD-1:0] vecIdx;
    logic [`INWD-1:0] dirVec [`INWD-1:0];

    // this value is shared among different sobol rngs to generate position of lsz
    cntWithEn u_cntWithEn(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .cntOut(cntNum)
        );

    LSZ u_LSZ(
        .in(cntNum),
        .lszIdx(vecIdx)
        );

    /* initialization of directional vectors for current dimension*/
    `ifdef INWD3
        always_comb begin : proc_dirVec_3
            dirVec[0] <= 'd4;
            dirVec[1] <= 'd2;
            dirVec[2] <= 'd1;
        end
    `endif

    `ifdef INWD4
        always_comb begin : proc_dirVec_4
            dirVec[0] <= 'd8;
            dirVec[1] <= 'd4;
            dirVec[2] <= 'd2;
            dirVec[3] <= 'd1;
        end
    `endif

    `ifdef INWD5
        always_comb begin : proc_dirVec_5
            dirVec[0] <= 'd16;
            dirVec[1] <= 'd8;
            dirVec[2] <= 'd4;
            dirVec[3] <= 'd2;
            dirVec[4] <= 'd1;
        end
    `endif

    `ifdef INWD6
        always_comb begin : proc_dirVec_6
            dirVec[0] <= 'd32;
            dirVec[1] <= 'd16;
            dirVec[2] <= 'd8;
            dirVec[3] <= 'd4;
            dirVec[4] <= 'd2;
            dirVec[5] <= 'd1;
        end
    `endif

    `ifdef INWD7
        always_comb begin : proc_dirVec_7
            dirVec[0] <= 'd64;
            dirVec[1] <= 'd32;
            dirVec[2] <= 'd16;
            dirVec[3] <= 'd8;
            dirVec[4] <= 'd4;
            dirVec[5] <= 'd2;
            dirVec[6] <= 'd1;
        end
    `endif

    `ifdef INWD8
        always_comb begin : proc_dirVec_8
            dirVec[0] <= 'd128;
            dirVec[1] <= 'd64;
            dirVec[2] <= 'd32;
            dirVec[3] <= 'd16;
            dirVec[4] <= 'd8;
            dirVec[5] <= 'd4;
            dirVec[6] <= 'd2;
            dirVec[7] <= 'd1;
        end
    `endif

    `ifdef INWD9
        always_comb begin : proc_dirVec_9
            dirVec[0] <= 'd256;
            dirVec[1] <= 'd128;
            dirVec[2] <= 'd64;
            dirVec[3] <= 'd32;
            dirVec[4] <= 'd16;
            dirVec[5] <= 'd8;
            dirVec[6] <= 'd4;
            dirVec[7] <= 'd2;
            dirVec[8] <= 'd1;
        end
    `endif

    `ifdef INWD10
        always_comb begin : proc_dirVec_10
            dirVec[0] <= 'd512;
            dirVec[1] <= 'd256;
            dirVec[2] <= 'd128;
            dirVec[3] <= 'd64;
            dirVec[4] <= 'd32;
            dirVec[5] <= 'd16;
            dirVec[6] <= 'd8;
            dirVec[7] <= 'd4;
            dirVec[8] <= 'd2;
            dirVec[9] <= 'd1;
        end
    `endif

    SobolRNG u_SobolRNG(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .vecIdx(vecIdx),
        .dirVec(dirVec),
        .out(sobolSeq)
        );

endmodule