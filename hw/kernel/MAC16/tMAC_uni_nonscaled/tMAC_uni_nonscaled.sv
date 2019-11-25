`include "SobolRNGDim1_8b.sv"
`include "tMUL_uni.sv"

module tMAC_uni_nonscaled (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input logic [7:0] iA [15:0],
    input logic [7:0] iB [15:0],
    input logic loadA,
    input logic loadB,
    output oC
);
    logic [7:0] sobolSeq;
    logic [7:0] sobolSeqC;
    logic [15:0] mulC;
    logic [7:0] cnt [15:0];
    logic [7:0] sum0 [3:0];
    logic [7:0] sum;
    logic [15:0] stop;
    
    SobolRNGDim1_8b U_SobolRNGDim1_8b(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .sobolSeq(sobolSeq)
        );

    SobolRNGDim1_8b U_SobolRNGDim1_8b_C(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .sobolSeq(sobolSeqC)
        );

    genvar i;
    generate
        for (i = 0; i < 16; i++) begin
            tMUL_uni U_tMUL_uni(
                .clk(clk),    // Clock
                .rst_n(rst_n),  // Asynchronous reset active low
                .iA(iA[i]),
                .iB(iB[i]),
                .loadA(loadA),
                .loadB(loadB),
                .sobolSeq(sobolSeq),
                .oC(mulC[i]),
                .stop(stop[i])
                );

            always_ff @(posedge clk or negedge rst_n) begin : proc_cnt_i
                if(~rst_n) begin
                    cnt[i] <= 0;
                end else begin
                    cnt[i] <= cnt[i] + mulC[i];
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin : proc_sum0
        if(~rst_n) begin
            sum0[0] <= 0;
            sum0[1] <= 0;
            sum0[2] <= 0;
            sum0[3] <= 0;
        end else begin
            sum0[0] <= cnt[0] + cnt[1] + cnt[2] + cnt[3];
            sum0[1] <= cnt[4] + cnt[5] + cnt[6] + cnt[7];
            sum0[2] <= cnt[8] + cnt[9] + cnt[10] + cnt[11];
            sum0[3] <= cnt[12] + cnt[13] + cnt[14] + cnt[15];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_sum
        if(~rst_n) begin
            sum <= 0;
        end else begin
            sum <= sum0[0] + sum0[1] + sum0[2] + sum0[3];
        end
    end

    assign oC = sum > sobolSeqC;

endmodule