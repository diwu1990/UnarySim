`include "skewedSync.sv"
`include "cordiv.sv"

module iscbdiv (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input sel,
    input dividend,
    input divisor,
    output quotient
);
    
    // in-stream correlation-based divisor
    // dividend / divisor = quotient
    // divisor is always larger than dividend

    // correlated bit stream after skewed synchronizer
    logic dividend_corr;
    logic divisor_corr;

    // input is send to a skewed synchronizer first.
    skewedSync #(
        .DEPTH(2)
        )
    U_skewedSync(
        .clk(clk),
        .rst_n(rst_n),
        .in({divisor,dividend}),
        .out({divisor_corr,dividend_corr})
        );

    cordiv U_cordiv(
        .clk(clk),
        .rst_n(rst_n),
        .srSel(sel),
        .dividend(dividend_corr),
        .divisor(divisor_corr),
        .quotient(quotient),
        .srOut()
        );

endmodule
