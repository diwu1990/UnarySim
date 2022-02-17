module fc3_MulXNOR # (
    parameter IDIM = 1,
    parameter FOLD = 1,
    parameter ODIM = 1
) (
    input logic iBit [IDIM - 1 : 0],
    input logic wBit [ODIM / FOLD * IDIM - 1 : 0],
    output logic oFmbs [ODIM / FOLD * IDIM - 1 : 0]
);
    genvar i;
    genvar j;

    generate 
    for (i = 0; i< ODIM/FOLD; i = i+1) begin
        for (j = 0; j<IDIM; j = j+1) begin
            assign oFmbs[i * IDIM + j] = ~(wBit[i * IDIM + j] ^ iBit[j]);
        end
    end
    endgenerate

    

endmodule