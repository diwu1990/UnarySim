module parallelCnt7 (
    input logic [6:0] in,
    output logic [2:0] out
);
    logic [1:0] tempSum [3:0];

    assign tempSum[0] = in[0] + in[1] + in[2];
    assign tempSum[1] = in[3] + in[4] + in[5];
    assign tempSum[2] = in[6] + tempSum[0][0] + tempSum[1][0];
    assign tempSum[3] = tempSum[0][1] + tempSum[1][1] + tempSum[2][1];

    assign out = {tempSum[3], tempSum[2][0]};
    
endmodule
