`include "HaltonSeq5Def.sv"
module HaltonSeq5 (
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
            out <= mcoutBin[0]+mcoutBin[1]+mcoutBin[2];
            // out <= mcoutBin[0]+mcoutBin[1]+mcoutBin[2]+mcoutBin[3];
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
            ModCnt5 U_ModCnt5(
                .clk(clk),
                .rst(rst),
                .cin(mccin[i]),
                .cout(mccout[i]),
                .out(mcout[i])
                );
        end
    endgenerate

    always_comb begin : proc_0
        mcoutBin[0] <= mcout[2];
        mcoutBin[1] <= mcout[1]*5;
        mcoutBin[2] <= mcout[0]*25;
    end

    // always_comb begin : proc_0
    //     mcoutBin[0] <= mcout[3];
    //     mcoutBin[1] <= mcout[2]*5;
    //     mcoutBin[2] <= mcout[1]*25;
    //     mcoutBin[2] <= mcout[0]*125;
    // end

endmodule