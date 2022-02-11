module fc3_Cmp #(
    parameter IWID = 10
) (
    input logic [IWID - 1 : 0] iData,
    input logic [IWID - 1 : 0] iRng,
    output logic oBit
);

    assign oBit = iData >= iRng;

endmodule