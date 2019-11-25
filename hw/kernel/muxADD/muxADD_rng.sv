`include "SobolRNGDim1.sv"
`include "muxADD.sv"

module muxADD_rng (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [`INUM-1:0] in,
    output logic out
);
    logic [`LOGINUM-1:0] sel;

    SobolRNGDim1 U_SobolRNGDim1(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .sobolSeq(sel)
        );

    muxADD U_muxADD(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(in),
        .sel(sel),
        .out(out)
        );

endmodule