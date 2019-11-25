module orADD (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [15:0] in,
    output logic out
);

    always_ff @(posedge clk or negedge rst_n) begin : proc_out
        if(~rst_n) begin
            out <= 0;
        end else begin
            out <= |in;
        end
    end

endmodule