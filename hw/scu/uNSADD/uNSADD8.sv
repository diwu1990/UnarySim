`include "parallelCnt7.sv"
// `define OFFSET 0
`define OFFSET 15

module uNSADD8 (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [7:0] in,
    output logic out
);
    
    logic [2:0] inCntLess;
    logic [3:0] tempSum;
    // logic [11:0] inAccBuf; // log2(N*L)+1
    // logic [12:0] offsetAccBuf; // log2(N*L)+1
    logic [9:0] theoryOneBuf; // log2(N*L)+1
    logic [9:0] actualOneBuf; // log2(N*L)+1

    parallelCnt7 U_parallelCnt7(
        .in(in[6:0]),
        .out(inCntLess)
        );

    assign tempSum = inCntLess + in[7];

    // always_ff @(posedge clk or negedge rst_n) begin : proc_accBuf
    //     if(~rst_n) begin
    //         inAccBuf <= 0;
    //         offsetAccBuf <= 0;
    //     end else begin
    //         inAccBuf <= inAccBuf + tempSum;
    //         offsetAccBuf <= offsetAccBuf + `OFFSET;
    //     end
    // end

    // assign theoryOneBuf = {inAccBuf, 1'b0} - offsetAccBuf;

    always_ff @(posedge clk or negedge rst_n) begin : proc_theoryOneBuf
        if(~rst_n) begin
            theoryOneBuf <= 0;
        end else begin
            theoryOneBuf <= theoryOneBuf + (tempSum << 1) - `OFFSET;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_actualOneBuf
        if(~rst_n) begin
            actualOneBuf <= 0;
        end else begin
            actualOneBuf <= actualOneBuf + {out, 1'b0};
        end
    end

    assign out = theoryOneBuf > actualOneBuf;

endmodule