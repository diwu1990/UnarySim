module Uni2Bi (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic in,
    output logic out
);

    // bipolar: output = (input + 1)/2
    logic [1:0] in_acc;
    logic [1:0] pc;

    assign pc = in + 1'b1;

    always_ff @(posedge clk or negedge rst_n) begin : proc_in_acc
        if(~rst_n) begin
            in_acc <= 2'b0;
        end else begin
            in_acc <= in_acc[0] + pc;
        end
    end

    // output a zero when in_acc overflows.
    assign out = in_acc[1];

endmodule