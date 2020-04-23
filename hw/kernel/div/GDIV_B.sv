module GDIV_B # (
    parameter BW=5
) (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [BW-1:0] randNum, 
    input logic dividend, 
    input logic divisor, 
    output logic quotient
);
    
    logic [BW-1:0] cnt;
    logic inc;
    logic dec;

    logic [2:0] xnorgate;
    logic [1:0] andgate;

    logic divisor_d;

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= {1'b1, {{BW-1}{1'b0}}};
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

    assign quotient = cnt > randNum;

    assign inc = andgate[0];
    assign dec = andgate[1];

    always_ff @(posedge clk or negedge rst_n) begin : proc_divisor_d
        if(~rst_n) begin
            divisor_d <= 0;
        end else begin
            divisor_d <= divisor;
        end
    end

    assign xnorgate[0] = ~(divisor ^ divisor_d);
    assign xnorgate[1] = ~(divisor ^ dividend);
    assign xnorgate[2] = ~(xnorgate[0] ^ ~quotient);

    assign andgate[0] = xnorgate[1] & xnorgate[2];
    assign andgate[1] = ~xnorgate[1] & ~xnorgate[2];

endmodule