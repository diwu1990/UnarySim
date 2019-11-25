`include "HaltonSeq3Def.sv"
module HaltonSeq3 (
    input logic clk,    // Clock
    input logic rst,  // Asynchronous reset active low
    output logic [`SEQWIDTH-1:0]out
);
    
    logic   [`DIGITWIDTH-1:0]mccin;
    logic   [`DIGITWIDTH-1:0]mccout;
    logic   [`LOGBASE-1:0]mcout[`DIGITWIDTH-1:0];
    logic   [`SEQWIDTH-1:0]mcoutBin[`DIGITWIDTH-1:0];

    always_ff @(posedge clk or posedge rst) begin : proc_out
        if(rst) begin
            out <= 0;
        end else begin
            // out <= mcoutBin[0]+mcoutBin[1]+mcoutBin[2];
            out <= mcoutBin[0]+mcoutBin[1]+mcoutBin[2]+mcoutBin[3]+mcoutBin[4];
        end
    end
    
    always_comb begin : proc_mccin0
        mccin[0] <= 1'b1;
    end

    genvar i;
    generate
        for (i = 1; i < `DIGITWIDTH; i++) begin
            always_comb begin : proc_cin
                mccin[i] <= mccout[i-1];
            end
        end
        for (i = 0; i < `DIGITWIDTH; i++) begin
            ModCnt3 U_ModCnt3(
                .clk(clk),
                .rst(rst),
                .cin(mccin[i]),
                .cout(mccout[i]),
                .out(mcout[i])
                );
        end
    endgenerate

    always_comb begin : proc_0
        mcoutBin[0] <= mcout[4];
        mcoutBin[1] <= mcout[3]*3;
        mcoutBin[2] <= mcout[2]*9;
        mcoutBin[3] <= mcout[1]*27;
        mcoutBin[4] <= mcout[0]*81;
    end

    // always_comb begin : proc_0
    //     mcoutBin[0] <= mcout[2];
    //     mcoutBin[1] <= mcout[1]*3;
    //     mcoutBin[2] <= mcout[0]*9;
    // end

endmodule