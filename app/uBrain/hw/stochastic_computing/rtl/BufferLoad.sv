module BufferLoad #(
    parameter IWID = 8,
    parameter IDIM = 4
) (
    input logic clk,
    input logic rst_n,
    input logic load,
    input logic [IWID - 1 : 0] iData [IDIM - 1 : 0],
    output logic [IWID - 1 : 0] oData [IDIM - 1 : 0]
);
    
    genvar i;
    generate
        for (i = 0; i < IDIM; i = i + 1) begin
            always @(posedge clk or negedge rst_n) begin
                if (~rst_n) begin
                    oData[i] <= 'b0;
                end
                else begin
                    oData[i] <= load ? iData[i] : oData[i];
                end
            end
        end
    endgenerate

endmodule