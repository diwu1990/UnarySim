`include "../../stream/shuffle/SkewedSync.sv"
`include "CORDIV_kernel.sv"

module ISCBDIV_U # (
    parameter DEP_KERNEL=2,
    parameter DEPLOG_KERNEL=1,
    parameter DEP_SYNC=2
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [DEPLOG-1:0] randNum,
    input logic dividend,
    input logic divisor,
    output logic quotient
);
    
    logic dividend_sync;
    logic divisor_sync;

    SkewedSync #(
        .DEP(DEP_SYNC)
    ) U_SkewedSync (
        .clk(clk),
        .rst_n(rst_n),
        .in0(dividend),
        .in1(divisor),
        .out0(dividend_sync),
        .out1(divisor_sync)
    );

    CORDIV_kernel #(
        .DEP(DEP_KERNEL),
        .DEPLOG(DEPLOG_KERNEL)
    ) U_CORDIV_kernel (
        .clk(clk),
        .rst_n(rst_n),
        .randNum(randNum),
        .dividend(dividend_sync),
        .divisor(divisor_sync),
        .quotient(quotient)
    );

endmodule