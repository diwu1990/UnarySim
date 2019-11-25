module muxADD (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic in,
    input logic sel,
    output logic out
);

    parameter INUM = 16;
    parameter LOGINUM = 4;

    logic [INUM-1:0] in;
    logic [LOGINUM-1:0] sel;

    always_ff @(posedge clk or negedge rst_n) begin : proc_out
        if(~rst_n) begin
            out <= 0;
        end else begin
            out <= in[sel];
        end
    end

endmodule