module Bi2Uni # (
    parameter DEP=3
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic in,
    output logic out
);

    // bipolar: output = 2 * input - 1
    logic [DEP-1:0] diff_acc;
    logic [1:0] pc;

    assign pc = {in, 1'b0};

    always_ff @(posedge clk or negedge rst_n) begin : proc_diff_acc
        if(~rst_n) begin
            diff_acc <= {1'b1, {{DEP-1}{1'b0}}};
        end else begin
            diff_acc <= diff_acc + pc - 1 - out;
        end
    end

    // as long as diff_acc is more than reset value, output a zero.
    assign out = diff_acc[DEP-1] & |diff_acc[DEP-2:0];

endmodule