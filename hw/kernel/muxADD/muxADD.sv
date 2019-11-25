`define INUM 8
`define LOGINUM 3

module muxADD (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [`INUM-1:0] in,
    input logic [`LOGINUM-1:0] sel,
    output logic out
);

    always_ff @(posedge clk or negedge rst_n) begin : proc_out
        if(~rst_n) begin
            out <= 0;
        end else begin
            out <= in[sel];
        end
    end

endmodule