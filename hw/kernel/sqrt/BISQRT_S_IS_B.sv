`include "../div/CORDIV_kernel.sv"

module BISQRT_S_IS_B #(
    parameter DEP=1,
    parameter DEPLOG=1
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [DEPLOG-1:0] randNum,
    input logic in,
    output logic out
);
    
    logic [1:0] mux;
    logic sel;
    logic trace;

    logic dividend;
    logic divisor;
    logic quotient;

    logic outUni;

    logic dff;
    logic dff_inv;

    assign dff_inv = ~dff;

    always_ff @(posedge clk or negedge rst_n) begin : proc_dff
        if(~rst_n) begin
            dff <= 0;
        end else begin
            dff <= dff_inv;
        end
    end

    assign mux[0] = in;
    assign mux[1] = 1;
    assign out = sel ? mux[1] : mux[0];
    assign sel = trace;

    assign trace = quotient;
    assign dividend = dff_inv & outUni;
    assign divisor = dff | dividend;

    CORDIV_kernel #(
        .DEP(DEP),
        .DEPLOG(DEPLOG)
    ) U_CORDIV_kernel(
        .clk(clk),
        .rst_n(rst_n),
        .randNum(randNum),
        .dividend(dividend),
        .divisor(divisor),
        .quotient(quotient)
    );

    Bi2Uni #(
        .DEP(3)
    ) U_Bi2Uni(
        .clk(clk),
        .rst_n(rst_n),
        .in(in),
        .out(outUni)
        );

endmodule
