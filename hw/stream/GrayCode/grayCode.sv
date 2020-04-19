`include "grayCodeDef.sv"

module grayCode (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    output logic [`GWIDTH-1:0]grayOutBin
    
);

    logic [`GWIDTH-1:0]cnt;

    always_ff @(posedge clk or negedge rst_n) begin : proc_1
        if(~rst_n) begin
            cnt <= 0;
        end else begin
            cnt <= cnt + 1;
        end
    end

    always_comb begin : proc_2
        grayOutBin <= (cnt>>1) ^ cnt;
    end

endmodule