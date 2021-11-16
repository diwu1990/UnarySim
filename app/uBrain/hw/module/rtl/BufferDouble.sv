module BufferDouble #(
    parameter IWID = 10,
    parameter OWID = IWID
) (
    input logic clk,
    input logic rst_n,
    input logic iAccSel,
    input logic iClear,
    input logic iHold,
    input logic [IWID - 1 : 0] iData,
    output logic [OWID - 1 : 0] oData
);

    logic [OWID - 1 : 0] reg0;
    logic [OWID - 1 : 0] reg1;
    logic [OWID - 1 : 0] iAdd;
    logic [OWID - 1 : 0] oAdd;

    // iAccSel == 0 means that reg0 is accumulated, and reg1 is output
    assign iAdd = iAccSel ? reg1 : reg0;
    assign oAdd = iClear ? 'b0 : (iAdd + iData);
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            reg0 <= 'b0;
            reg1 <= 'b0;
        end
        else begin
            reg0 <= (iAccSel |  iHold) ? reg0 : oAdd;
            reg1 <= (iAccSel & ~iHold) ? oAdd : reg1;
        end
    end
    assign oData = iAccSel ? reg0 : reg1;

endmodule