`include "SobolRNGDim1.sv"
`define DATAWD `INWD 

module uMUL_bi (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic iA,
    input logic [`DATAWD-1:0] iB,
    input logic loadB,
    output logic oC
);
    
    logic [`DATAWD-1:0] iB_buf;
    logic [`DATAWD-1:0] sobolSeq;
    logic [`DATAWD-1:0] sobolSeq_reverse;

    always_ff @(posedge clk or negedge rst_n) begin : proc_iB_buf
        if(~rst_n) begin
            iB_buf <= 0;
        end else begin
            if(loadB) begin
                iB_buf <= iB;
            end else begin
                iB_buf <= iB_buf;
            end
        end
    end

    SobolRNGDim1 U_SobolRNGDim1(
        .clk(clk),
        .rst_n(rst_n),
        .enable(iA),
        .sobolSeq(sobolSeq)
        );

    SobolRNGDim1 U_SobolRNGDim1_reverse(
        .clk(clk),
        .rst_n(rst_n),
        .enable(~iA),
        .sobolSeq(sobolSeq_reverse)
        );

    always_comb begin : proc_oC
        oC <= (iA & (iB_buf > sobolSeq)) | (~iA & (iB_buf <= sobolSeq_reverse));
    end

endmodule