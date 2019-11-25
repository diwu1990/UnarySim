`include "uSADD16.sv"
`include "uMUL_bi.sv"

module uMAC_bi_scaled (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input logic [15:0] iA,
    input logic [7:0] iB [15:0],
    input logic loadB,
    output oC
);
    logic [15:0] mulC;
    
    genvar i;
    generate
        for (i = 0; i < 16; i++) begin
            uMUL_bi U_uMUL_bi(
                .clk(clk),    // Clock
                .rst_n(rst_n),  // Asynchronous reset active low
                .iA(iA[i]),
                .iB(iB[i]),
                .loadB(loadB),
                .oC(mulC[i])
                );
        end
    endgenerate

    uSADD16 U_uSADD16(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(mulC),
        .out(oC)
        );

endmodule