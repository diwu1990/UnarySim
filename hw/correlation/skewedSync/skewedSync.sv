module skewedSync (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input [1:0] in,
    output [1:0] out
);
    parameter DEPTH = 2;

    // bit array of index 1 has a higher value than that of index 0

    logic [1:0] out;

    logic [DEPTH - 1 : 0] cnt;
    logic cntFull;
    logic cntEmpty;

    assign cntFull = &cnt;
    assign cntEmpty = ~|cnt;
    assign out[1] = in[1];

    always_comb begin : proc_out0
        if(in[0] == in[1]) begin
            out[0] = in[0];
        end else begin
            if(in[0]) begin
                out[0] = cntFull ? 1 : 0;
            end else begin
                out[0] = cntEmpty ? 0 : 1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= 0;
        end else begin
            cnt <= cntFull ? cnt : (cnt + in[0]);
        end
    end

endmodule