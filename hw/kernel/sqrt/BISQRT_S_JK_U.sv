`include "../jkff/JKFF.sv"

module BISQRT_S_JK_U (
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

endmodule
