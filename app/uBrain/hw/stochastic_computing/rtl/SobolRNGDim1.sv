`include "SobolRNG.sv"
`include "LSZ.sv"
`include "CntEn.sv"

module SobolRNGDim1 #(
    parameter RWID = 8,
    parameter RWL2 = $clog2(RWID)
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic enable,
    output logic [RWID-1:0]sobolSeq
);

    logic [RWID - 1 : 0] cntNum;
    logic [RWL2 - 1 : 0] vecIdx;
    logic [RWID - 1 : 0] dirVec [RWID - 1 : 0];

    // this value is shared among different sobol rngs to generate position of lsz
    CntEn #(
        .CWID(RWID)
    ) u_CntEn(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .cnt(cntNum)
        );

    LSZ #(
        .IWID(RWID)
    ) u_LSZ(
        .in(cntNum),
        .lszIdx(vecIdx)
        );

    /* initialization of directional vectors for current dimension*/
    generate
        case (RWID)
            3 : always_comb begin : proc_dirVec_3
                    dirVec[0] <= 'd4;
                    dirVec[1] <= 'd2;
                    dirVec[2] <= 'd1;
                end
            4 : always_comb begin : proc_dirVec_4
                    dirVec[0] <= 'd8;
                    dirVec[1] <= 'd4;
                    dirVec[2] <= 'd2;
                    dirVec[3] <= 'd1;
                end
            5 : always_comb begin : proc_dirVec_5
                    dirVec[0] <= 'd16;
                    dirVec[1] <= 'd8;
                    dirVec[2] <= 'd4;
                    dirVec[3] <= 'd2;
                    dirVec[4] <= 'd1;
                end
            6 : always_comb begin : proc_dirVec_6
                    dirVec[0] <= 'd32;
                    dirVec[1] <= 'd16;
                    dirVec[2] <= 'd8;
                    dirVec[3] <= 'd4;
                    dirVec[4] <= 'd2;
                    dirVec[5] <= 'd1;
                end
            7 : always_comb begin : proc_dirVec_7
                    dirVec[0] <= 'd64;
                    dirVec[1] <= 'd32;
                    dirVec[2] <= 'd16;
                    dirVec[3] <= 'd8;
                    dirVec[4] <= 'd4;
                    dirVec[5] <= 'd2;
                    dirVec[6] <= 'd1;
                end
            8 : always_comb begin : proc_dirVec_8
                    dirVec[0] <= 'd128;
                    dirVec[1] <= 'd64;
                    dirVec[2] <= 'd32;
                    dirVec[3] <= 'd16;
                    dirVec[4] <= 'd8;
                    dirVec[5] <= 'd4;
                    dirVec[6] <= 'd2;
                    dirVec[7] <= 'd1;
                end
            9 : always_comb begin : proc_dirVec_9
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
            10 : always_comb begin : proc_dirVec_10
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
        endcase
    endgenerate
        
    SobolRNG #(
        .RWID(RWID)
    ) u_SobolRNG(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .vecIdx(vecIdx),
        .dirVec(dirVec),
        .out(sobolSeq)
        );

endmodule