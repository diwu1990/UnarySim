`include "SobolRNGDim1.sv"
`define DATAWD `INWD 

module dMUL_bi (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [`DATAWD-1:0] iA,
    input logic [`DATAWD-1:0] iB,
    input logic loadA,
    input logic loadB,
    output logic oC
);
    
    logic [`DATAWD-1:0] iA_buf;
    logic [`DATAWD-1:0] iB_buf;
    logic [`DATAWD-1:0] sobolSeqA;
    logic [`DATAWD-1:0] sobolSeqB;
    logic [`DATAWD-1:0] cntA;
    logic [`DATAWD-1:0] cntB;
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_cntA
        if(~rst_n) begin
            cntA <= 0;
        end else begin
            cntA <= cntA + 1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_cntB
        if(~rst_n) begin
            cntB <= 0;
        end else begin
            cntB <= cntB + (&cntA);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_iA_buf
        if(~rst_n) begin
            iA_buf <= 0;
        end else begin
            if(loadA) begin
                iA_buf <= iA;
            end else begin
                iA_buf <= iA_buf;
            end
        end
    end

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

    SobolRNGDim1 U_SobolRNGDim1_A(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .sobolSeq(sobolSeqA)
        );

    SobolRNGDim1 U_SobolRNGDim1_B(
        .clk(clk),
        .rst_n(rst_n),
        .enable(&cntB),
        .sobolSeq(sobolSeqB)
        );

    always_comb begin : proc_oC
        oC <= ~((iA_buf > sobolSeqA) ^ (iB_buf > sobolSeqB));
    end

endmodule