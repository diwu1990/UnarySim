module CntEn #(
    parameter CWID = 8
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic enable,
    output logic [CWID - 1 : 0] cnt
);

    always @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            cnt <= 'b0;
        end else begin
            cnt <= cnt + enable;
        end
    end

endmodule