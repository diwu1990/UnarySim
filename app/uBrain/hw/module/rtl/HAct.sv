module HAct #(
    parameter IWID = 16,
    parameter ADIM = 32, // accumulation depth
    parameter OWID = 8,
    parameter PZER = ADIM * (2**OWID) / 2,
    parameter PPON = ADIM * (2**OWID) / 2 + (2**OWID) / 2,
    parameter PNON = ADIM * (2**OWID) / 2 - (2**OWID) / 2,
    parameter RELU = 0
) (
    input logic [IWID - 1 : 0] iData,
    output logic [OWID - 1 : 0] oData
);

    generate
        if (RELU == 1) begin
            always_comb begin : relu
                if (iData >= PPON) begin
                    oData <= {OWID{1'b1}};
                end
                else if (iData <= PZER) begin
                    oData <= {1'b1, {(OWID-1){1'b0}}};
                end
                else begin
                    oData <= iData[OWID - 1 : 0] + {1'b1, {(OWID-1){1'b0}}};
                end
            end
        end
        else if (RELU == 2) begin
            always_comb begin : sigmoid
                if (iData >= PPON) begin
                    oData <= {OWID{1'b1}};
                end
                else if (iData <= PNON) begin
                    oData <= {OWID{1'b0}};
                end
                else begin
                    oData <= {1'b1, iData[OWID - 1 : 1]};
                end
            end
        end
        else begin
            always_comb begin : hardtanh
                if (iData >= PPON) begin
                    oData <= {OWID{1'b1}};
                end
                else if (iData <= PNON) begin
                    oData <= {OWID{1'b0}};
                end
                else begin
                    oData <= iData[OWID - 1 : 0] + {1'b1, {(OWID-1){1'b0}}};
                end
            end
        end
    endgenerate

endmodule