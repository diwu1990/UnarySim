module relu (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input in,
    output out    
);
    parameter DEPTH = 4;

    logic overHalf;
    logic [DEPTH:0]cnt;
    logic inc;
    logic dec;

    assign inc = in;
    assign dec = ~in;

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= {1,{(DEPTH-1){1'b0}}};
        end else begin
            if(inc & ~dec & ~&cnt) begin
                cnt <= cnt + 1;
            end else if(~inc & dec & ~|cnt) begin
                cnt <= cnt - 1;
            end else begin
                cnt <= cnt;
            end
        end
    end

    assign overHalf = cnt[DEPTH-1];

    assign out = overHalf ? in : 0;

endmodule