`include "parallelCnt7.sv"

module uSADD8 (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [7:0] in,
    output logic out
);
    
    logic [2:0] inCntLess;
    logic [3:0] tempSum;
    logic [3:0] accumulation;
    logic [2:0] accBuf;

    parallelCnt7 U_parallelCnt7(
        .in(in[6:0]),
        .out(inCntLess)
        );

    assign tempSum = inCntLess + in[7];

    assign accumulation = accBuf + tempSum[2:0];

    always_ff @(posedge clk or negedge rst_n) begin : proc_accBuf
        if(~rst_n) begin
            accBuf <= 0;
        end else begin
            accBuf <= accumulation[2:0];
        end
    end

    assign out = tempSum[3] ? tempSum[3] : accumulation[3];

endmodule