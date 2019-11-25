`include "parallelCnt7.sv"

module parallelCnt15 (
    input logic [14:0] in,
    output logic [3:0] out
);
    
    logic [2:0] tempSum7 [1:0];
    logic [1:0] tempSum [2:0];

    parallelCnt7 U_parallelCnt7_0(
        .in(in[6:0]),
        .out(tempSum7[0])
        );

    parallelCnt7 U_parallelCnt7_1(
        .in(in[13:7]),
        .out(tempSum7[1])
        );

    assign tempSum[0] = in[14] + tempSum7[0][0] + tempSum7[1][0];
    assign tempSum[1] = tempSum[0][1] + tempSum7[0][1] + tempSum7[1][1];
    assign tempSum[2] = tempSum[1][1] + tempSum7[0][2] + tempSum7[1][2];

    assign out = {tempSum[2], tempSum[1][0], tempSum[0][0]};
    
endmodule
