module GSQRT_B # (
    parameter DEP=5
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [DEP-1:0] randNum, 
    input logic in, 
    output logic out
);
    
    logic [DEP-1:0] cnt;
    logic inc;
    logic dec;
    logic out_d1;

    logic cntFull;
    logic cntEmpty;

    assign cntFull = &cnt;
    assign cntEmpty = ~(|cnt);
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= {1'b1, {{DEP-1}{1'b0}}};
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

    assign out = cnt > randNum;
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_out_d1
        if(~rst_n) begin
            out_d1 <= 0;
        end else begin
            out_d1 <= out;
        end
    end

    assign inc = in;
    assign dec = ~(out ^ out_d1);

endmodule