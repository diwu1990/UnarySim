`define DATAWD 8

module bADD (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [`DATAWD-1:0] iA,
    input logic [`DATAWD-1:0] iB,
    output logic [`DATAWD:0] oC
);
    logic [`DATAWD-1:0] iA_buf;
    logic [`DATAWD-1:0] iB_buf;

    always_ff @(posedge clk or negedge rst_n) begin : proc_oC
        if(~rst_n) begin
            oC <= 0;
            iA_buf <= 0;
            iB_buf <= 0;
        end else begin
            iA_buf <= iA;
            iB_buf <= iB;
            oC <= iA_buf + iB_buf;
        end
    end

endmodule