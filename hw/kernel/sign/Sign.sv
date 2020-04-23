module Sign # (
    parameter DEP=3
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic value, 
    output logic sign
);
    
    logic [DEP-1:0] cnt;

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= {1'b1, {{DEP-1}{1'b0}}};
        end else begin
            if(value & ~&cnt) begin
                cnt <= cnt + 1;
            end else if(~value & ~|cnt) begin
                cnt <= cnt - 1;
            end else begin
                cnt <= cnt;
            end
        end
    end

    assign sign = ~cnt[DEP-1];

endmodule