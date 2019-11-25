`include "SobolRNGDef.sv"
`include "SobolRNGtwo.sv"
`include "LSZ.sv"
`include "grayCode.sv"

module SobolRNGnine (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    output logic [`INWD-1:0]sobolSeq[8:0]
);

    logic [`INWD-1:0] grayNum;
    logic [`LOGINWD-1:0]vecIdx0;
    logic [`LOGINWD-1:0]vecIdx1;

    logic [`LOGINWD-1:0]vecIdx[1:0];

    // this value is shared among different sobol rngs to generate position of lsz
    grayCode u_grayCode(
        .clk()
        .rst_n(rst_n)
        .grayOut(grayNum)
        );

    LSZ u_LSZ(
        .in(grayNum),
        .outoh(),
        .lszidx(vecIdx0),
        .outohr(),
        .lszidxr(vecIdx1),
        );

    assign vecIdx[0] = vecIdx0;
    assign vecIdx[1] = vecIdx1;

    logic [`INWD:0]dirVec01[`INWD-1:0];
    logic [`INWD:0]dirVec23[`INWD-1:0];
    logic [`INWD:0]dirVec45[`INWD-1:0];
    logic [`INWD:0]dirVec67[`INWD-1:0];
    logic [`INWD:0]dirVec89[`INWD-1:0];

    logic [`INWD-1:0]out01[1:0];
    logic [`INWD-1:0]out23[1:0];
    logic [`INWD-1:0]out45[1:0];
    logic [`INWD-1:0]out67[1:0];
    logic [`INWD-1:0]out89[1:0];

    /* initialization of directional vectors for each dimension*/
    // dim 0 and 1
    dirVec01[0] = 9'd1;
    dirVec01[1] = 9'd1;
    dirVec01[2] = 9'd1;
    dirVec01[3] = 9'd1;
    dirVec01[4] = 9'd1;
    dirVec01[5] = 9'd1;
    dirVec01[6] = 9'd1;
    dirVec01[7] = 9'd1;
    // dim 2 and 3
    dirVec23[0] = 9'd335;
    dirVec23[1] = 9'd451;
    dirVec23[2] = 9'd381;
    dirVec23[3] = 9'd493;
    dirVec23[4] = 9'd283;
    dirVec23[5] = 9'd415;
    dirVec23[6] = 9'd341;
    dirVec23[7] = 9'd511;
    // dim 4 and 5
    dirVec45[0] = 9'd397;
    dirVec45[1] = 9'd509;
    dirVec45[2] = 9'd499;
    dirVec45[3] = 9'd175;
    dirVec45[4] = 9'd115;
    dirVec45[5] = 9'd349;
    dirVec45[6] = 9'd197;
    dirVec45[7] = 9'd295;
    // dim 6 and 7
    dirVec67[0] = 9'd429;
    dirVec67[1] = 9'd493;
    dirVec67[2] = 9'd111;
    dirVec67[3] = 9'd63;
    dirVec67[4] = 9'd151;
    dirVec67[5] = 9'd475;
    dirVec67[6] = 9'd101;
    dirVec67[7] = 9'd179;
    // dim 8 and 9
    dirVec89[0] = 9'd265;
    dirVec89[1] = 9'd473;
    dirVec89[2] = 9'd227;
    dirVec89[3] = 9'd291;
    dirVec89[4] = 9'd157;
    dirVec89[5] = 9'd463;
    dirVec89[6] = 9'd175;
    dirVec89[7] = 9'd87;

    SobolRNGtwo u_SobolRNGtwo_01(
        .clk(clk),
        .rst_n(clk),
        .vecIdx(vecIdx),
        .dirVec(dirVec01),
        .out(out01)
        );

    SobolRNGtwo u_SobolRNGtwo_23(
        .clk(clk),
        .rst_n(clk),
        .vecIdx(vecIdx),
        .dirVec(dirVec23),
        .out(out23)
        );

    SobolRNGtwo u_SobolRNGtwo_45(
        .clk(clk),
        .rst_n(clk),
        .vecIdx(vecIdx),
        .dirVec(dirVec45),
        .out(out45)
        );

    SobolRNGtwo u_SobolRNGtwo_67(
        .clk(clk),
        .rst_n(clk),
        .vecIdx(vecIdx),
        .dirVec(dirVec67),
        .out(out67)
        );

    SobolRNGtwo u_SobolRNGtwo_89(
        .clk(clk),
        .rst_n(clk),
        .vecIdx(vecIdx),
        .dirVec(dirVec89),
        .out(out89)
        );

    assign sobolSeq[0] = out01[1];
    assign sobolSeq[1] = out23[0];
    assign sobolSeq[2] = out23[1];
    assign sobolSeq[3] = out45[0];
    assign sobolSeq[4] = out45[1];
    assign sobolSeq[5] = out67[0];
    assign sobolSeq[6] = out67[1];
    assign sobolSeq[7] = out89[0];
    assign sobolSeq[8] = out89[1];

endmodule