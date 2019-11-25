`include "SobolRNGDef.sv"

module SobolRNG (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic enable,
    input logic [`LOGINWD-1:0] vecIdx, // choose dirVec according to vecIdx
    input logic [`INWD-1:0] dirVec [`INWD-1:0],
    output logic [`INWD-1:0] out // output random number
);

    always_ff @(posedge clk or negedge rst_n) begin : proc_out
        if(~rst_n) begin
            out <= 0;
        end else begin
            if(enable) begin
                out <= out ^ dirVec[vecIdx];
            end else begin
                out <= out;
            end
        end
    end

endmodule