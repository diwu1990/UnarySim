`include "SobolRNGDim1.sv"
`include "muxADD.sv"

module dADD (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [`INUM-1:0] in,
    output logic out
);
    logic [`LOGINUM-1:0] sel;

    logic [`LOGINUM-1:0] cnt;

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= 0;
        end else begin
            cnt <= cnt + 1;
        end
    end

    SobolRNGDim1 U_SobolRNGDim1(
        .clk(clk),
        .rst_n(rst_n),
        .enable(cnt == 1),
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