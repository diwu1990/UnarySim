module SkewedSyncInt # (
    parameter DEP=2,
    parameter BWIN=1,
    parameter BWOUT=2
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [BWIN-1:0] in0,
    input logic in1,
    output logic [DEP-1:0] out0,
    output logic out1
);

    logic [DEP-1:0] cnt;
    logic cntFull;
    logic cntEmpty;
    
    logic [DEP:0] sum;
    logic sum_overflow;
    logic [DEP-1:0] sum_clip;

    logic [DEP:0] sum_res;
    logic sum_res_overflow;
    logic [DEP-1:0] sum_res_clip;

    assign cntFull = &cnt;
    assign cntEmpty = ~|cnt;

    assign sum = cnt + in0;
    assign sum_overflow = sum[DEP];
    assign sum_clip = sum_overflow ? {{DEP-1}{1'b1}} : sum[DEP-1:0];

    assign sum_res = sum - out0;
    assign sum_res_overflow = sum_res[DEP];
    assign sum_res_clip = sum_res_overflow ? {{DEP-1}{1'b1}} : sum[DEP-1:0];

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= 0;
        end else begin
            cnt <= sum_res_clip;
        end
    end

    always_comb begin : proc_out0
        if (in1) begin
            out0 <= sum_clip;
        end else begin
            out0 <= 0;
        end
    end

    assign out1 = in1;

endmodule