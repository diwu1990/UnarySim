module JKFF (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic J,
    input logic K,
    output logic out
);

    always_ff @(posedge clk or negedge rst_n) begin : proc_out
        if(~rst_n) begin
            out <= 0;
        end else begin
            case ({J,K})
                2'b00: out <= out;
                2'b10: out <= 1'b1;
                2'b01: out <= 1'b0;
                2'b11: out <= ~out;
                default : out <= 0;
            endcase
        end
    end

endmodule