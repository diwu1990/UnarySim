module fc3_RELU #(
    parameter IDIM = 256, 
    parameter IWID = $clog2(110*32) + 1 + 10,
    parameter ADIM = 110*32, // accumulation depth
    parameter ODIM = IDIM,
    parameter OWID = 10,
    parameter PZER = ADIM * (2**OWID) / 2,
    parameter PPON = ADIM * (2**OWID) / 2 + (2**OWID) / 2,
    parameter PNON = ADIM * (2**OWID) / 2 - (2**OWID) / 2
    //parameter RELU = 1
) (
    input logic [IWID - 1 : 0] iData [IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oData [ODIM - 1 : 0]
);

    genvar i;
    generate
        for (i = 0; i < IDIM; i = i + 1) begin
            //if (RELU == 1) begin
                always_comb begin : relu
                    if (iData[i] >= PPON) begin
                        oData[i] <= {OWID{1'b1}};
                    end
                    else if (iData[i] <= PZER) begin
                        oData[i] <= {1'b1, {(OWID-1){1'b0}}};
                    end
                    else begin
                        oData[i] <= iData[i][OWID - 1 : 0] + {1'b1, {(OWID-1){1'b0}}};
                    end
                end
            //end
        end
    endgenerate

endmodule