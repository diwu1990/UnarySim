module MuxArray #(
    parameter IDIM = 8,
    parameter IWID = 8,
    parameter FOLD = 2,
    parameter SWID = ($clog2(FOLD) < 1) ? 1 : $clog2(FOLD),
    parameter ODIM = IDIM / FOLD,
    parameter OWID = IWID
) (
    input logic [IWID - 1 : 0] iData [IDIM - 1 : 0],
    input logic [SWID - 1 : 0] iSel,
    output logic [OWID - 1 : 0] oData [ODIM - 1 : 0]
);

    logic [IWID - 1 : 0] tData [IDIM - 1 : 0];
    logic [FOLD - 1 : 0] tSel;

    genvar i, j;
    generate
        for (i = 0; i < FOLD; i = i + 1) begin
            assign tSel[i] = (iSel == i);
            for (j = 0; j < ODIM; j = j + 1) begin
                assign tData[ODIM * i + j] = tSel[i] ? iData[ODIM * i + j] : 'b0;
            end
        end
        
        for (i = 0; i < ODIM; i = i + 1) begin
            always_comb begin
                oData[i] = 'b0;
                for (int k = 0; k < FOLD; k = k + 1) begin
                    oData[i] = oData[i] | tData[ODIM * k + i];
                end
            end
        end
        
    endgenerate
    
endmodule