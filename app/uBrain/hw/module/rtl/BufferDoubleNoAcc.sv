module BufferDoubleNoAcc #(
    parameter IWID = 10,
    parameter OWID = IWID
) (
    input logic clk,
    input logic rst_n,
    input logic iSel,
    input logic [IWID - 1 : 0] iData,
    output logic [OWID - 1 : 0] oData
);

    logic [OWID - 1 : 0] reg0;
    logic [OWID - 1 : 0] reg1;

    // iSel == 0 means that reg0 is loaded, and reg1 is output
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            reg0 <= 'b0;
            reg1 <= 'b0;
        end
        else begin
            reg0 <= iSel ? reg0 : iData;
            reg1 <= iSel ? iData : reg1;
        end
    end
    assign oData = iSel ? reg0 : reg1;

endmodule