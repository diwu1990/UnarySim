`include "parallelCnt15.sv"

module uSADD16 (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [15:0] in,
    output logic out
);
    
    logic [3:0] inCntLess;
    logic [4:0] tempSum;
    logic [4:0] accumulation;
    logic [3:0] accBuf;

    parallelCnt15 U_parallelCnt15(
        .in(in[14:0]),
        .out(inCntLess)
        );

    assign tempSum = inCntLess + in[15];

    assign accumulation = accBuf + tempSum[3:0];

    always_ff @(posedge clk or negedge rst_n) begin : proc_accBuf
        if(~rst_n) begin
            accBuf <= 0;
        end else begin
            accBuf <= accumulation[3:0];
        end
    end

    assign out = tempSum[4] ? tempSum[4] : accumulation[4];

endmodule