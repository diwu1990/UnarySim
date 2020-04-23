`include "../../stream/shuffle_int/SkewedSyncInt.sv"
`include "../div/CORDIV_kernel.sv"

module BISQRT_O_U #(
    parameter DEP_SYNC=2,
    parameter DEP_KERNEL=1,
    parameter DEPLOG_KERNEL=1,
    parameter DEP_EMIT=3,
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [DEPLOG_KERNEL-1:0] randNum,
    input logic in,
    output logic out
);
    
    logic dividend;
    logic divisor;
    logic [DEP_SYNC-1:0] dividend_sync;
    logic divisor_sync;
    logic [DEP_SYNC-1:0] quotient;

    logic [DEP_EMIT-1:0] acc_emit;
    logic [DEP_EMIT:0] sum;
    logic sum_overflow;
    logic [DEP_EMIT-1:0] sum_clip;
    logic [DEP_EMIT:0] temp;
    logic temp_non_0;

    assign dividend = ~out;
    assign divisor = out;

    SkewedSyncInt # (
        .DEP(DEP_SYNC)
    ) U_SkewedSyncInt (
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in0(dividend), // 1-Pin
        .in1(divisor), // Pin
        .out0(dividend_sync),
        .out1(divisor_sync)
    );

    genvar i;
    generate
        for (i = 0; i < DEP_SYNC; i++) begin
            CORDIV_kernel #(
                .DEP(DEP_KERNEL),
                .DEPLOG(DEPLOG_KERNEL)
            ) U_CORDIV_kernel (
                .clk(clk),
                .rst_n(rst_n),
                .randNum(randNum),
                .dividend(dividend_sync[i]),
                .divisor(divisor_sync),
                .quotient(quotient[i])
            );
        end
    endgenerate

    assign temp = acc_emit + quotient;
    assign temp_non_0 = |temp;
    assign out = (~in) & temp_non_0;

    assign sum = temp - out;
    assign sum_overflow = sum[DEP_EMIT];
    assign sum_clip = sum_overflow ? {{DEP_EMIT}{1'b1}} : sum[DEP_EMIT-1:0];

    always_ff @(posedge clk or negedge rst_n) begin : proc_acc_emit
        if(~rst_n) begin
            acc_emit <= 0;
        end else begin
            acc_emit <= sum_clip;
        end
    end

endmodule