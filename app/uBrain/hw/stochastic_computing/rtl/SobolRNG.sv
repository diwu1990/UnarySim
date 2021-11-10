module SobolRNG #(
    parameter RWID = 8,
    parameter RWL2 = $clog2(RWID)
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic enable,
    input logic [RWL2 - 1 : 0] vecIdx, // choose dirVec according to vecIdx
    input logic [RWID - 1 : 0] dirVec [RWID - 1 : 0],
    output logic [RWID - 1 : 0] out // output random number
);

    always @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            out <= 'b0;
        end else begin
            if(enable) begin
                out <= out ^ dirVec[vecIdx];
            end else begin
                out <= out;
            end
        end
    end

endmodule