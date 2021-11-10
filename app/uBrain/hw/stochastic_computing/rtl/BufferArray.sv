module BufferArray #(
    parameter IDIM = 16,
    parameter IWID = 1,
    parameter ODIM = IDIM,
    parameter OWID = 32
) (
    input logic clk,
    input logic rst_n,
    input logic iAccSel,
    input logic iClear,
    input logic [IWID - 1 : 0] iData [IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oData [ODIM - 1 : 0]
);

    logic [OWID - 1 : 0] reg0 [ODIM - 1 : 0];
    logic [OWID - 1 : 0] reg1 [ODIM - 1 : 0];
    logic [OWID - 1 : 0] iAdd [ODIM - 1 : 0];
    logic [OWID - 1 : 0] oAdd [ODIM - 1 : 0];

    genvar i;
    generate
        for (i = 0; i < IDIM; i = i + 1) begin
            // iAccSel == 0 means that reg0 is accumulated, and reg1 is output
            assign iAdd[i] = iAccSel ? reg1[i] : reg0[i];
            assign oAdd[i] = iClear ? 'b0 : (iAdd[i] + iData[i]);
            always @(posedge clk or negedge rst_n) begin
                if (~rst_n) begin
                    reg0[i] <= 'b0;
                    reg1[i] <= 'b0;
                end
                else begin
                    reg0[i] <= iAccSel ? reg0[i] : oAdd[i];
                    reg1[i] <= iAccSel ? oAdd[i] : reg1[i];
                end
            end
            assign oData[i] = iAccSel ? reg0[i] : reg1[i];
        end
    endgenerate

endmodule