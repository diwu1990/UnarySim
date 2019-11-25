`include "param_def.sv"

module MAC (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [`MAC_BW-1 : 0] iA,
    input logic [`MAC_BW-1 : 0] iB,
    input logic [2*`MAC_BW+3 : 0] iC,
    input logic acc_en,
    // output logic [`MAC_BW-1 : 0] oA,
    // output logic [`MAC_BW-1 : 0] oB,
    output logic [2*`MAC_BW+3 : 0] oC
);
    
    logic [2*`MAC_BW+3 : 0] iC_mac;

    always_comb begin : proc_iC_mac
        iC_mac <= acc_en ? oC : iC
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_output
        if(~rst_n) begin
            oA <= 0;
            // oB <= 0;
            // oC <= 0;
        end else begin
            // oA <= iA;
            // oB <= iB;
            oC <= iA * iB + iC_mac;
        end
    end

endmodule