module Dff #(
    parameter IWID = 1
) (
    input logic clk,
    input logic rst_n,
    input logic [IWID - 1 : 0] in,
    output logic [IWID - 1 : 0] out
);

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            out <= 'b0;
        end else begin
            out <= in;
        end
    end
    
endmodule