module GDIV_U # (
    parameter DEP=5
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [DEP-1:0] randNum, 
    input logic dividend, 
    input logic divisor, 
    output logic quotient
);
    
    logic [DEP-1:0] cnt;
    logic inc;
    logic dec;

    logic cntFull;
    logic cntEmpty;

    assign cntFull = &cnt;
    assign cntEmpty = ~(|cnt);

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= {1'b1, {{DEP-1}{1'b0}}};
        end else begin
            if(value & ~cntFull) begin
                cnt <= cnt + 1;
            end else if(~value & ~cntEmpty) begin
                cnt <= cnt - 1;
            end else begin
                cnt <= cnt;
            end
        end
    end

    assign quotient = cnt > randNum;

    assign inc = dividend;
    assign dec = quotient & divisor;

endmodule