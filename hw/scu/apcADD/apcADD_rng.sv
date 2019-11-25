`include "SobolRNGDim1.sv"
`include "apcADD.sv"

module apcADD_rng (
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

    apcADD U_apcADD(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(in),
        .randNum(sel),
        .out(out)
        );

endmodule