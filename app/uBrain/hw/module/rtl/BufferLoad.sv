module BufferLoad #(
    parameter IWID = 10
) (
    input logic clk,
    input logic rst_n,
    input logic load,
    input logic [IWID - 1 : 0] iData,
    output logic [IWID - 1 : 0] oData
);
    
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            oData <= 'b0;
        end
        else begin
            oData <= load ? iData : oData;
        end
    end

endmodule