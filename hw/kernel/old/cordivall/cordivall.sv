`include "cordiv.sv"

module cordivall (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    // control port
    input [7 : 0]randNum,
    input sel,
    // data port
    input dividend,
    input divisor,
    output quotient
);
    
    logic [7 : 0] dividend_cnt;
    logic [7 : 0] divisor_cnt;
    logic dividend_regen;
    logic divisor_regen;

    always_ff @(posedge clk or negedge rst_n) begin : proc_dividend_cnt
        if(~rst_n) begin
            dividend_cnt <= 8'h80;
        end else begin
            dividend_cnt <= dividend ? dividend_cnt + 1 : dividend_cnt - 1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_divisor_cnt
        if(~rst_n) begin
            divisor_cnt <= 8'h80;
        end else begin
            divisor_cnt <= divisor ? divisor_cnt + 1 : divisor_cnt - 1;
        end
    end

    assign dividend_regen = (dividend_cnt >= randNum) ? 1 : 0;
    assign divisor_regen = (divisor_cnt >= randNum) ? 1 : 0;

    cordiv U_cordiv(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        // control port
        .srSel(sel),
        // data port
        .dividend(dividend_regen),
        .divisor(divisor_regen),
        .quotient(quotient),
        .srOut()
        );



endmodule