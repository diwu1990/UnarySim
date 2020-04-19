`define INUM 8
`define LOGINUM 3

module apcADD (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [`INUM-1:0] in,
    input logic [`LOGINUM-1:0] randNum,
    output logic out
);
    
    // 16 input
    // logic [7:0] temp;
    // logic [1:0] tempSum [3:0];

    // logic [`LOGINUM-1:0] sum;

    // assign temp[0] = in[0] | in[1];
    // assign temp[1] = in[2] | in[3];
    // assign temp[2] = in[4] | in[5];
    // assign temp[3] = in[6] | in[7];
    // assign temp[4] = in[8] | in[9];
    // assign temp[5] = in[10] | in[11];
    // assign temp[6] = in[12] | in[13];
    // assign temp[7] = in[14] | in[15];

    // assign tempSum[0] = temp[0] + temp[1] + temp[2];
    // assign tempSum[1] = temp[3] + temp[4] + temp[5];
    // assign tempSum[2] = temp[6] + tempSum[0][0] + tempSum[1][0];
    // assign tempSum[3] = tempSum[0][1] + tempSum[1][1] + tempSum[2][1];

    // always_ff @(posedge clk or negedge rst_n) begin : proc_sum
    //     if(~rst_n) begin
    //         sum <= 0;
    //     end else begin
    //         sum <= {tempSum[3], tempSum[2][0], 1'b0} + temp[7];
    //     end
    // end

    // assign out = sum > randNum;

    // 8 input
    logic [3:0] temp;
    logic [1:0] tempSum;

    logic [`LOGINUM-1:0] sum;

    assign temp[0] = in[0] | in[1];
    assign temp[1] = in[2] | in[3];
    assign temp[2] = in[4] | in[5];
    assign temp[3] = in[6] | in[7];

    assign tempSum = temp[0] + temp[1] + temp[2];

    always_ff @(posedge clk or negedge rst_n) begin : proc_sum
        if(~rst_n) begin
            sum <= 0;
        end else begin
            sum <= {tempSum, 1'b0} + temp[3];
        end
    end

    assign out = sum > randNum;

endmodule