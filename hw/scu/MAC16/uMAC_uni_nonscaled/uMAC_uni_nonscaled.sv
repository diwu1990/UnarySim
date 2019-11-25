`include "uNSADD16.sv"
`include "uMUL_uni.sv"

module uMAC_uni_nonscaled (
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
            uMUL_uni U_uMUL_uni(
                .clk(clk),    // Clock
                .rst_n(rst_n),  // Asynchronous reset active low
                .iA(iA[i]),
                .iB(iB[i]),
                .loadB(loadB),
                .oC(mulC[i])
                );
        end
    endgenerate

    uNSADD16 U_uNSADD16(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .in(mulC),
        .out(oC)
        );

endmodule