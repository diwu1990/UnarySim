`include "../jkff/JKFF.sv"
`include "../../stream/shuffle/Bi2Uni.sv"

module BISQRT_S_JK_B # (
    parameter DEP=3
) (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input in,
    output out
);
    
    logic [1:0] mux;
    logic sel;
    logic trace;

    logic Jport;
    logic Kport;
    logic JKout;

    assign mux[0] = in;
    assign mux[1] = 1;
    assign out = sel ? mux[1] : mux[0];
    assign sel = trace;

    assign trace = JKout;
    assign Jport = out;
    assign Kport = 1;

    JKFF U_JKFF(
        .clk(clk),
        .rst_n(rst_n),
        .J(Jport),
        .K(Kport),
        .out(JKout)
        );

    Bi2Uni #(
        .DEP(DEP)
    ) U_Bi2Uni(
        .clk(clk),
        .rst_n(rst_n),
        .in(in),
        .out(out)
        );

endmodule
