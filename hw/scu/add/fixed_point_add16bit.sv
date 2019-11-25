module fixed_point_add16bit (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input [15:0] a,
    input [15:0] b,
    output logic [16:0] c
);
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_c
        if(~rst_n) begin
            c <= 0;
        end else begin
            c <= a+b;
        end
    end

endmodule