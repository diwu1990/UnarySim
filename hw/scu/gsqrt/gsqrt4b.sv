module gsqrt4b (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input [3:0]randNum,
    input in,
    output out
);
    
    logic [3:0] cnt;
    logic inc;
    logic dec;
    logic out_d1;
    logic cntFull;
    logic cntEmpty;

    assign cntFull = &cnt;
    assign cntEmpty = ~|cnt;
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= 4'b1000;
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

    assign out = cnt >= randNum;
    always_ff @(posedge clk or negedge rst_n) begin : proc_out_d1
        if(~rst_n) begin
            out_d1 <= 0;
        end else begin
            out_d1 <= out;
        end
    end

    assign inc = in;
    assign dec = out & out_d1;

endmodule