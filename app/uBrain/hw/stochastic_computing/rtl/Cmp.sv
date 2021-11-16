module Cmp #(
    parameter IWID = 8
) (
    input logic [IWID - 1 : 0] iData,
    input logic [IWID - 1 : 0] iRng,
    output logic oBit
);

    assign oBit = iData >= iRng;

endmodule