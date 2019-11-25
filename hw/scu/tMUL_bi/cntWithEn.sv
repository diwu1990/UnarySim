`include "SobolRNGDef.sv"

module cntWithEn (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic enable,
    output logic [`CNTWD-1:0]cntOut
);

    logic [`CNTWD-1:0]cnt;

    always_ff @(posedge clk or negedge rst_n) begin : proc_1
        if(~rst_n) begin
            cnt <= 0;
        end else begin
            cnt <= cnt + enable;
        end
    end

    always_comb begin : proc_2
        // grayOut <= (cnt>>1) ^ cnt;
        cntOut <= cnt;
    end

endmodule