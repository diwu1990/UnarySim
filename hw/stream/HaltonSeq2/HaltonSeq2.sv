`include "HaltonSeq2Def.sv"
module HaltonSeq2 (
    input logic clk,    // Clock
    input logic rst,  // Asynchronous reset active low
    output logic [`SEQWIDTH-1:0]out
);
    
    logic   [`DIGITWIDTH-1:0]mccin;
    logic   [`DIGITWIDTH-1:0]mccout;
    logic   [`LOGBASE-1:0]mcout[`DIGITWIDTH-1:0];

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
            ModCnt2 U_ModCnt2(
                .clk(clk),
                .rst(rst),
                .cin(mccin[i]),
                .cout(mccout[i]),
                .out(mcout[i])
                );
        end
    endgenerate

    generate
        for (i = 0; i < `SEQWIDTH; i++) begin
            always_comb begin : proc_out
                out[i] <= mcout[`SEQWIDTH-1-i];
            end
        end
    endgenerate
    
endmodule