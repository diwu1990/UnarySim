`include "SobolSeq2Def.sv"
module LSZ2 (
    input logic [`INWIDTH-1:0]in,
    output logic [`LOGINWIDTH-1:0]out
);

    // priority based
    logic [`INWIDTH-1:0]onehot;
    logic [`INWIDTH-1:0]inacc;

    genvar i;

    assign inacc[0] = ~in[0];
    generate
        for (i = 1; i < `INWIDTH; i++) begin
            assign inacc[i] = inacc[i-1] | ~in[i];
        end
    endgenerate

    assign onehot[0] = inacc[0];
    generate
        for (i = 1; i < `INWIDTH; i++) begin
            assign onehot[i] = inacc[i-1] ^ inacc[i];
        end
    endgenerate

    always_comb begin : proc_
        case(onehot)
            'b0 : out = `INWIDTH+1;
            'b1 : out = 'd1;
            'b10 : out = 'd2;
            'b100 : out = 'd3;
            'b1000 : out = 'd4;
            'b10000 : out = 'd5;
            'b100000 : out = 'd6;
            'b1000000 : out = 'd7;
            'b10000000 : out = 'd8;
            default : out = 'd0;
        endcase // onehot
    end

    // lut based


endmodule