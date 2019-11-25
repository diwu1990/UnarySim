`include "fulladder.sv"

module apcadd16 (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input [3:0] randNum,
    input [15:0] in,
    output out
);
    
    logic [3:0] orGate;
    logic [3:0] andGate;

    genvar i;
    generate
        for (int i = 0; i < 4; i++) begin
            orGate[i] = in[4*i] | in[4*i+1];
            andGate[i] = in[4*i+2] & in[4*i+3];
        end
    endgenerate

    logic [3:0] sum;
    logic [3:0] co;

    fulladder U_fulladder0(
        .a(orGate[0]),
        .b(andGate[0]),
        .ci(orGate[1]),
        .sum(sum[0]),
        .co(co[0])
        );

    fulladder U_fulladder1(
        .a(andGate[1]),
        .b(orGate[2]),
        .ci(andGate[2]),
        .sum(sum[1]),
        .co(co[1])
        ); 

    fulladder U_fulladder2(
        .a(sum[0]),
        .b(sum[1]),
        .ci(orGate[3]),
        .sum(sum[2]),
        .co(co[2])
        ); 

    fulladder U_fulladder3(
        .a(co[0]),
        .b(co[1]),
        .ci(co[2]),
        .sum(sum[3]),
        .co(co[3])
        ); 

    logic [3:0] cnt;

    assign cnt[0] = andGate[3];
    assign cnt[1] = sum[2];
    assign cnt[2] = sum[3];
    assign cnt[3] = co[3];

    assign out = (cnt >= randNum);

endmodule