`include "HaltonSeq3Def.sv"

module ModCnt3 (
    input logic clk,    // Clock
    input logic rst,  // Asynchronous reset active low
    input logic cin,
    output logic cout,
    output logic [`LOGBASE-1:0]out
);

    logic [`LOGBASE-1:0]cnt;
    logic cntmax;

    assign cntmax = (cnt == `BASE-1);
    assign cout = cntmax & cin;
    assign out = cnt;

    always_ff @(posedge clk or posedge rst) begin : proc_out
        if(rst) begin
            cnt <= 0;
        end else begin
            cnt <= cout ? (cnt == 0) : (cnt + cin);
        end
    end

endmodule