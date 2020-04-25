module BISQRT_O_U #(
    parameter DEP=3
    parameter DEP_SR=4
    parameter DEPLOG_SR=2
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [DEPLOG_SR-1:0] randNum,
    input logic in,
    output logic out
);
    
    logic out_inv;
    logic [DEP_SR-1:0] sr;
    logic sr_out;
    logic emit_out;
    logic [DEP-1:0] diff_acc;
    logic [1:0] pc;

    assign out_inv = ~out;

    // following code is for shift reg
    assign sr_out = sr[randNum];

    always_ff @(posedge clk or negedge rst_n) begin : proc_sr
        if(~rst_n) begin
            for (int i = 0; i < DEP; i++) begin
                sr[i] <= i%2;
            end
        end else begin
            sr[DEP-1] <= out_inv;
            for (int i = 0; i < DEP-1; i++) begin
                sr[i] <= sr[i+1];
            end
        end
    end

    assign emit_out = out & sr_out;

    // following code is for non-scaled add
    assign pc = in + emit_out;

    always_ff @(posedge clk or negedge rst_n) begin : proc_diff_acc
        if(~rst_n) begin
            diff_acc <= {1'b1, {{DEP-1}{1'b0}}};
        end else begin
            diff_acc <= diff_acc + pc - out;
        end
    end

    // as long as diff_acc is more than reset value, output a zero.
    assign out = diff_acc[DEP-1] & |diff_acc[DEP-2:0];

endmodule