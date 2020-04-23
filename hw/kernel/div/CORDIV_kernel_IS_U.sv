`include "CORDIV_kernel.sv"

module CORDIV_kernel_IS_U #(
    parameter BW=8,
    parameter DEP=2,
    parameter DEPLOG=1
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [BW-1:0] randNum,
    input logic dividend,
    input logic divisor,
    output logic quotient
);
    
    logic [BW-1:0] dividend_cnt;
    logic [BW-1:0] divisor_cnt;
    logic dividend_regen;
    logic divisor_regen;

    always_ff @(posedge clk or negedge rst_n) begin : proc_dividend_cnt
        if(~rst_n) begin
            dividend_cnt <= 0;
        end else begin
            dividend_cnt <= dividend ? dividend_cnt + 1 : dividend_cnt - 1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_divisor_cnt
        if(~rst_n) begin
            divisor_cnt <= 0;
        end else begin
            divisor_cnt <= divisor ? divisor_cnt + 1 : divisor_cnt - 1;
        end
    end

    assign dividend_regen = (dividend_cnt > randNum) ? 1 : 0;
    assign divisor_regen = (divisor_cnt > randNum) ? 1 : 0;

    CORDIV_kernel #(
        .DEP(DEP),
        .DEPLOG(DEPLOG)
    ) U_CORDIV_kernel(
        .clk(clk),
        .rst_n(rst_n),
        .randNum(randNum),
        .dividend(dividend_regen),
        .divisor(divisor_regen),
        .quotient(quotient)
    );

endmodule
