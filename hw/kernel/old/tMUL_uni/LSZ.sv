`include "SobolRNGDef.sv"
module LSZ (
    input logic [`INWD-1:0] in, // input gray number
    output logic [`LOGINWD-1:0] lszIdx // index of least siginificant 0
);

    // priority based
    logic [`INWD-1:0] inacc;
    logic [`INWD-1:0] outoh;

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

    `ifdef INWD3
        always_comb begin : proc_3
            case(outoh)
                'b001 : lszIdx = 'd0;
                'b010 : lszIdx = 'd1;
                'b100 : lszIdx = 'd2;
                default : lszIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD4
        always_comb begin : proc_4
            case(outoh)
                'b0001 : lszIdx = 'd0;
                'b0010 : lszIdx = 'd1;
                'b0100 : lszIdx = 'd2;
                'b1000 : lszIdx = 'd3;
                default : lszIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD5
        always_comb begin : proc_5
            case(outoh)
                'b00001 : lszIdx = 'd0;
                'b00010 : lszIdx = 'd1;
                'b00100 : lszIdx = 'd2;
                'b01000 : lszIdx = 'd3;
                'b10000 : lszIdx = 'd4;
                default : lszIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD6
        always_comb begin : proc_6
            case(outoh)
                'b000001 : lszIdx = 'd0;
                'b000010 : lszIdx = 'd1;
                'b000100 : lszIdx = 'd2;
                'b001000 : lszIdx = 'd3;
                'b010000 : lszIdx = 'd4;
                'b100000 : lszIdx = 'd5;
                default : lszIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD7
        always_comb begin : proc_7
            case(outoh)
                'b0000001 : lszIdx = 'd0;
                'b0000010 : lszIdx = 'd1;
                'b0000100 : lszIdx = 'd2;
                'b0001000 : lszIdx = 'd3;
                'b0010000 : lszIdx = 'd4;
                'b0100000 : lszIdx = 'd5;
                'b1000000 : lszIdx = 'd6;
                default : lszIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD8
        always_comb begin : proc_8
            case(outoh)
                'b00000001 : lszIdx = 'd0;
                'b00000010 : lszIdx = 'd1;
                'b00000100 : lszIdx = 'd2;
                'b00001000 : lszIdx = 'd3;
                'b00010000 : lszIdx = 'd4;
                'b00100000 : lszIdx = 'd5;
                'b01000000 : lszIdx = 'd6;
                'b10000000 : lszIdx = 'd7;
                default : lszIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD9
        always_comb begin : proc_9
            case(outoh)
                'b000000001 : lszIdx = 'd0;
                'b000000010 : lszIdx = 'd1;
                'b000000100 : lszIdx = 'd2;
                'b000001000 : lszIdx = 'd3;
                'b000010000 : lszIdx = 'd4;
                'b000100000 : lszIdx = 'd5;
                'b001000000 : lszIdx = 'd6;
                'b010000000 : lszIdx = 'd7;
                'b100000000 : lszIdx = 'd8;
                default : lszIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD10
        always_comb begin : proc_10
            case(outoh)
                'b0000000001 : lszIdx = 'd0;
                'b0000000010 : lszIdx = 'd1;
                'b0000000100 : lszIdx = 'd2;
                'b0000001000 : lszIdx = 'd3;
                'b0000010000 : lszIdx = 'd4;
                'b0000100000 : lszIdx = 'd5;
                'b0001000000 : lszIdx = 'd6;
                'b0010000000 : lszIdx = 'd7;
                'b0100000000 : lszIdx = 'd8;
                'b1000000000 : lszIdx = 'd9;
                default : lszIdx = 'd0;
            endcase // onehot
        end
    `endif

endmodule