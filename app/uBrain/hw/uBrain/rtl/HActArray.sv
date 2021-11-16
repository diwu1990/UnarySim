module HActArray #(
    parameter IDIM = 4,
    parameter IWID = 16,
    parameter ADIM = 32, // accumulation depth
    parameter ODIM = IDIM,
    parameter OWID = 8,
    parameter PZER = ADIM * (2**OWID) / 2,
    parameter PPON = ADIM * (2**OWID) / 2 + (2**OWID) / 2,
    parameter PNON = ADIM * (2**OWID) / 2 - (2**OWID) / 2,
    parameter RELU = 0
) (
    input logic clk,
    input logic rst_n,
    input logic [IWID - 1 : 0] iData [IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oData [ODIM - 1 : 0]
);

    genvar i;
    generate
        for (i = 0; i < IDIM; i = i + 1) begin
            if (RELU == 1) begin
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
            end
            else if (RELU == 2) begin
                always_comb begin : sigmoid
                    if (iData[i] >= PPON) begin
                        oData[i] <= {OWID{1'b1}};
                    end
                    else if (iData[i] <= PNON) begin
                        oData[i] <= {OWID{1'b0}};
                    end
                    else begin
                        oData[i] <= {1'b1, iData[i][OWID - 1 : 1]};
                    end
                end
            end
            else begin
                always_comb begin : hardtanh
                    if (iData[i] >= PPON) begin
                        oData[i] <= {OWID{1'b1}};
                    end
                    else if (iData[i] <= PNON) begin
                        oData[i] <= {OWID{1'b0}};
                    end
                    else begin
                        oData[i] <= iData[i][OWID - 1 : 0] + {1'b1, {(OWID-1){1'b0}}};
                    end
                end
            end
        end
    endgenerate

endmodule