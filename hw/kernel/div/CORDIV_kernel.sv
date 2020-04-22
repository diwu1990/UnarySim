module CORDIV_kernel #(
    parameter DEP=2,
    parameter DEPLOG=1
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [DEPLOG-1:0] randNum,
    input logic dividend,
    input logic divisor,
    output logic quotient
);
    
    logic [DEP-1:0] sr;
    logic [1:0] mux;
    logic sel;

    // when DEP=2 or DEP=1, both DEPLOG sbould be 1.
    assign mux[0] = sr[randNum];
    assign mux[1] = dividend;
    assign sel = divisor;

    assign quotient = sel ? mux[1] : mux[0];

    always_ff @(posedge clk or negedge rst_n) begin : proc_sr
        if(~rst_n) begin
            for (int i = 0; i < DEP; i++) begin
                sr[i] <= i%2;
            end
        end else begin
            sr[DEP-1] <= quotient;
            for (int i = 0; i < DEP-1; i++) begin
                sr[i] <= sr[i+1];
            end
        end
    end

endmodule
