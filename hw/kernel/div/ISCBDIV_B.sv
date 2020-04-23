`include "../../stream/shuffle/Uni2Bi.sv"
`include "../../stream/shuffle/Bi2Uni.sv"
`include "../abs/Abs.sv"
`include "../sign/Sign.sv"
`include "ISCBDIV_U.sv"

module ISCBDIV_B # (
    parameter DEP_ABS=3,
    parameter DEP_KERNEL=2,
    parameter DEPLOG_KERNEL=1,
    parameter DEP_SYNC=2
) (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input logic [DEPLOG_KERNEL-1:0] randNum,
    input logic dividend,
    input logic divisor,
    output logic quotient
);
    
    logic dividend_sign;
    logic dividend_abs;
    logic dividend_abs_uni;

    logic divisor_sign;
    logic divisor_abs;
    logic divisor_abs_uni;

    logic quotient_abs_uni;
    logic quotient_abs;

    Abs #(
        .DEP(DEP_ABS)
    ) U_Abs_dividend (
        .clk(clk),
        .rst_n(rst_n),
        .value(dividend),
        .sign(dividend_sign),
        .abs(dividend_abs)
    );

    Abs #(
        .DEP(DEP_ABS)
    ) U_Abs_divisor (
        .clk(clk),
        .rst_n(rst_n),
        .value(divisor),
        .sign(divisor_sign),
        .abs(divisor_abs)
    );

    Bi2Uni #(
        .DEP(3)
    ) U_Bi2Uni_dividend (
        .clk(clk),
        .rst_n(rst_n),
        .in(dividend_abs),
        .out(dividend_abs_uni)
    );

    Bi2Uni #(
        .DEP(3)
    ) U_Bi2Uni_divisor (
        .clk(clk),
        .rst_n(rst_n),
        .in(divisor_abs),
        .out(divisor_abs_uni)
    );

    ISCBDIV_U #(
        .DEP_KERNEL(DEP_KERNEL),
        .DEPLOG_KERNEL(DEPLOG_KERNEL),
        .DEP_SYNC(DEP_SYNC)
    ) U_ISCBDIV_U (
        .clk(clk),
        .rst_n(rst_n),
        .randNum(randNum),
        .dividend(dividend_abs_uni),
        .divisor(divisor_abs_uni),
        .quotient(quotient_abs_uni)
    );

    Uni2Bi U_Uni2Bi (
        .clk(clk),
        .rst_n(rst_n),
        .in(quotient_abs_uni),
        .out(quotient_abs)
    );

    assign quotient = quotient_abs ^ dividend_sign ^ divisor_sign;

endmodule