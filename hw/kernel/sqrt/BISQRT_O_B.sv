`include "../../stream/shuffle/Uni2Bi.sv"
`include "../../stream/shuffle/Bi2Uni.sv"
`include "BISQRT_O_U.sv"

module BISQRT_O_B #(
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
    
    logic in_uni;
    logic out_uni;

    Bi2Uni #(
        .DEP(3)
    ) U_Bi2Uni_dividend (
        .clk(clk),
        .rst_n(rst_n),
        .in(in),
        .out(in_uni)
    );

    BISQRT_O_U #(
        .DEP_SYNC(DEP_SYNC),
        .DEP_KERNEL(DEP_KERNEL),
        .DEPLOG_KERNEL(DEPLOG_KERNEL),
        .DEP_EMIT(DEP_EMIT)
    ) (
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .randNum(randNum),
        .in(in_uni),
        .out(out_uni)
    );

    Uni2Bi U_Uni2Bi (
        .clk(clk),
        .rst_n(rst_n),
        .in(out_uni),
        .out(out)
    );

endmodule