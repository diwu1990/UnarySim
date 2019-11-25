`include "SobolRNGDef.sv"

module SobolRNGtwo (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input logic [`LOGINWD-1:0]vecIdx[1:0],
    input logic [`INWD:0]dirVec[`INWD-1:0],
    output logic [`INWD-1:0]out[1:0]
);
    // vecIdx[1] is the reverse of vecIdx[0]
    // one SobolRNG can generate two sobol sequences.
    // it receives a merged directional vector array for two sobol sequences from input dirVec.
    // the merged array is 8 * 9bit

    logic [`INWD-1:0]vecReg[1:0][`INWD-1:0];

    // printf("Example of 8 bits:\n");
    // printf("s0v0_0 s0v1_1 s0v2_2 s0v3_3 s0v4_4 s0v5_5 s0v6_6 s0v7_7\n");
    // printf("s1v7_7 s0v1_0 s0v2_1 s0v3_2 s0v4_3 s0v5_4 s0v6_5 s0v7_6\n");
    // printf("s1v7_6 s1v6_6 s0v2_0 s0v3_1 s0v4_2 s0v5_3 s0v6_4 s0v7_5\n");
    // printf("s1v7_5 s1v6_5 s1v5_5 s0v3_0 s0v4_1 s0v5_2 s0v6_3 s0v7_4\n");
    // printf("s1v7_4 s1v6_4 s1v5_4 s1v4_4 s0v4_0 s0v5_1 s0v6_2 s0v7_3\n");
    // printf("s1v7_3 s1v6_3 s1v5_3 s1v4_3 s1v3_3 s0v5_0 s0v6_1 s0v7_2\n");
    // printf("s1v7_2 s1v6_2 s1v5_2 s1v4_2 s1v3_2 s1v2_2 s0v6_0 s0v7_1\n");
    // printf("s1v7_1 s1v6_1 s1v5_1 s1v4_1 s1v3_1 s1v2_1 s1v1_1 s0v7_0\n");
    // printf("s1v7_0 s1v6_0 s1v5_0 s1v4_0 s1v3_0 s1v2_0 s1v1_0 s1v0_0\n");
    
    always_comb begin : proc_vecReg
        vecReg[0][0] = {dirVec[0][8],7'b0};
        vecReg[0][1] = {dirVec[1][8:7],6'b0};
        vecReg[0][2] = {dirVec[2][8:6],5'b0};
        vecReg[0][3] = {dirVec[3][8:5],4'b0};
        vecReg[0][4] = {dirVec[4][8:4],3'b0};
        vecReg[0][5] = {dirVec[5][8:3],2'b0};
        vecReg[0][6] = {dirVec[6][8:2],1'b0};
        vecReg[0][7] = {dirVec[7][8:1]};

        vecReg[1][0] = {dirVec[7][0],7'b0};
        vecReg[1][1] = {dirVec[6][1:0],6'b0};
        vecReg[1][2] = {dirVec[5][2:0],5'b0};
        vecReg[1][3] = {dirVec[4][3:0],4'b0};
        vecReg[1][4] = {dirVec[3][4:0],3'b0};
        vecReg[1][5] = {dirVec[2][5:0],2'b0};
        vecReg[1][6] = {dirVec[1][6:0],1'b0};
        vecReg[1][7] = {dirVec[0][7:0]};
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_out
        if(~rst_n) begin
            out[0] <= 0;
            out[1] <= 0;
        end else begin
            out[0] <= out[0] ^ vecReg[0][vecIdx[0]];
            out[1] <= out[1] ^ vecReg[1][vecIdx[1]];
        end
    end

endmodule