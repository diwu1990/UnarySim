module fc3_MulDecomp # (
    parameter IDIM = 1,
    parameter FOLD = 1,
    parameter ODIM = 1
) (
    input logic iBit [IDIM - 1 : 0],
    input logic wBit [ODIM / FOLD * IDIM - 1 : 0],
    output logic oFmbs [ODIM / FOLD * IDIM - 1 : 0],
    output logic enable [ODIM / FOLD * IDIM - 1 : 0]
);
    genvar i;
    genvar j;

    generate 
    for (i = 0; i< ODIM/FOLD; i = i+1) begin
        for (j = 0; j<IDIM; j = j+1) begin
            assign enable[i * IDIM + j] = ~iBit[j];
            assign oFmbs[i * IDIM + j] = (wBit[i * IDIM + j] & ~enable[i * IDIM + j]) | (~wBit[i * IDIM + j] & enable[i * IDIM + j]);
        end
    end
    endgenerate

    

endmodule