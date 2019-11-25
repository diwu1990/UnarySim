module exp (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input [2:0] randBit,
    input in,
    output out
);

    logic [2:0] d;
    logic [3:0] addOut;
    logic [1:0] mulOut;

    // add0
    assign addOut[0] = randBit[0] ? 0 : in;

    // add1
    assign addOut[1] = randBit[1] ? 1 : addOut[0];

    // add2
    assign addOut[2] = randBit[0] ? 1 : in;

    // add3
    always_ff @(posedge clk or negedge rst_n) begin : proc_d2
        if(~rst_n) begin
            d[2] <= 0;
        end else begin
            d[2] <= mulOut[1];
        end
    end
    assign addOut[3] = randBit[2] ? d[2] : addOut[2];

    // mul0
    always_ff @(posedge clk or negedge rst_n) begin : proc_d0
        if(~rst_n) begin
            d[0] <= 0;
        end else begin
            d[0] <= in;
        end
    end
    assign mulOut[0] = d[0] & addOut[0];

    // mul1
    always_ff @(posedge clk or negedge rst_n) begin : proc_d1
        if(~rst_n) begin
            d[1] <= 0;
        end else begin
            d[1] <= mulOut[0];
        end
    end
    assign mulOut[1] = d[1] & addOut[1];

    assign out = addOut[3];

endmodule