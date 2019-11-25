`include "SobolRNGDef.sv"
module LSZ (
    input logic [`INWD-1:0]in,
    output logic [`INWD-1:0]outoh,
    output logic [`LOGINWD-1:0]lszidx,
    output logic [`INWD-1:0]outohr,
    output logic [`LOGINWD-1:0]lszidxr
);

    // priority based
    logic [`INWD-1:0]inacc;

    genvar i;

    assign inacc[0] = ~in[0];
    generate
        for (i = 1; i < `INWD; i++) begin
            assign inacc[i] = inacc[i-1] | ~in[i];
        end
    endgenerate

    assign outoh[0] = inacc[0];
    generate
        for (i = 1; i < `INWD; i++) begin
            assign outoh[i] = inacc[i-1] ^ inacc[i];
        end
    endgenerate

    generate
        for (int i = 0; i < `INWD; i++) begin
            assign outohr[i] = outoh[`INWD-1-i];
        end
    endgenerate

    always_comb begin : proc_
        case(outoh)
            'b00000000 : begin lszidx = 'd7; lszidxr = 'd0;end
            'b00000001 : begin lszidx = 'd0; lszidxr = 'd7;end
            'b00000010 : begin lszidx = 'd1; lszidxr = 'd6;end
            'b00000100 : begin lszidx = 'd2; lszidxr = 'd5;end
            'b00001000 : begin lszidx = 'd3; lszidxr = 'd4;end
            'b00010000 : begin lszidx = 'd4; lszidxr = 'd3;end
            'b00100000 : begin lszidx = 'd5; lszidxr = 'd2;end
            'b01000000 : begin lszidx = 'd6; lszidxr = 'd1;end
            'b10000000 : begin lszidx = 'd7; lszidxr = 'd0;end
            default : begin lszidx = 'd0; lszidxr = 'd7;end
        endcase // onehot
    end


endmodule