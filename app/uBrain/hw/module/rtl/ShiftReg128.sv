module ShiftReg128 #(
    parameter SRDP = 128
) (
    input logic clk,
    input logic rst_n,
    input logic in,
    output logic [SRDP - 1 : 0] out
);

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            out <= 'b0;
        end else begin
            out <= {out[SRDP - 2 : 0], in};
        end
    end
    
endmodule