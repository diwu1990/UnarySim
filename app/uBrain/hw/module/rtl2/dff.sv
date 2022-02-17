module dff # (
    parameter BW = 10,
    parameter WIDTH = 1,
    parameter HEIGHT = 1
)(
    input logic clk,
    input logic rstn,
    input logic [BW-1:0] d [WIDTH*HEIGHT-1:0],
    output logic [BW-1:0] q [WIDTH*HEIGHT-1:0]
);
    genvar i;
    generate 
    for (i = 0; i < WIDTH*HEIGHT; i = i + 1) begin
    always @ (posedge clk or negedge rstn) begin
        if (!rstn) begin
            q[i] <= 0;
	end
        else begin
            q[i] <= d[i];
        end
    end
    end
    endgenerate
endmodule