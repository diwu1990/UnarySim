module gdiv5b (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input [4:0]randNum,
    input dividend,
    input divisor,
    output quotient
);
    
    logic [4:0] cnt;
    logic inc;
    logic dec;
    logic cntFull;
    logic cntEmpty;

    assign cntFull = &cnt;
    assign cntEmpty = ~|cnt;

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= 5'b10000;
        end else begin
            if(inc & ~dec & ~cntFull) begin
                cnt <= cnt + 1;
            end else if(~inc & dec & ~cntEmpty) begin
                cnt <= cnt - 1;
            end else begin
                cnt <= cnt;
            end
        end
    end

    assign quotient = (cnt >= randNum);

    assign inc = dividend;
    assign dec = quotient & divisor;

endmodule
