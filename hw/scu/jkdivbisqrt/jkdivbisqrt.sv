module jkdivbisqrt (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input in,
    output out
);
    
    logic JKout;
    logic sel;
    logic [1:0] mux;
    logic Jport;
    logic Kport;

    assign mux[0] = 1;
    assign mux[1] = in;
    assign out = sel ? mux[1] : mux[0];
    assign sel = JKout;
    assign Kport = out;
    assign Jport = 1;

    always_ff @(posedge clk or negedge rst_n) begin : proc_JKout
            if(~rst_n) begin
                JKout <= 1;
            end else begin
                JKout <= Kport ? ~JKout : Jport;
            end
        end    

endmodule
