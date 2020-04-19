`define INWD3

module SobolRNGDim1_3b_10b (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic enable,
    output logic sobolSeq
);
    
    `ifdef INWD3
        parameter INWD = 3;
        parameter LOGINWD = 2;
    `endif

    `ifdef INWD4
        parameter INWD = 4;
        parameter LOGINWD = 2;
    `endif

    `ifdef INWD5
        parameter INWD = 5;
        parameter LOGINWD = 3;
    `endif

    `ifdef INWD6
        parameter INWD = 6;
        parameter LOGINWD = 3;
    `endif

    `ifdef INWD7
        parameter INWD = 7;
        parameter LOGINWD = 3;
    `endif

    `ifdef INWD8
        parameter INWD = 8;
        parameter LOGINWD = 3;
    `endif

    `ifdef INWD9
        parameter INWD = 9;
        parameter LOGINWD = 4;
    `endif

    `ifdef INWD10
        parameter INWD = 10;
        parameter LOGINWD = 4;
    `endif

    logic [INWD-1:0]sobolSeq;
    logic [LOGINWD-1:0] vecIdx;
    logic [INWD-1:0] dirVec [INWD-1:0];

    // binary counter
    logic [INWD-1:0]cnt;
    always_ff @(posedge clk or negedge rst_n) begin : proc_1
        if(~rst_n) begin
            cnt <= 0;
        end else begin
            cnt <= cnt + enable;
        end
    end

    // least significant zero index
    logic [INWD-1:0] inacc;
    logic [INWD-1:0] outoh;

    genvar i;

    assign inacc[0] = ~cnt[0];
    generate
        for (i = 1; i < INWD; i++) begin
            assign inacc[i] = inacc[i-1] | ~cnt[i];
        end
    endgenerate

    assign outoh[0] = inacc[0];
    generate
        for (i = 1; i < INWD; i++) begin
            assign outoh[i] = inacc[i-1] ^ inacc[i];
        end
    endgenerate

    // vector lookup and sequence generation
    always_ff @(posedge clk or negedge rst_n) begin : proc_sobolSeq
        if(~rst_n) begin
            sobolSeq <= 0;
        end else begin
            if(enable) begin
                sobolSeq <= sobolSeq ^ dirVec[vecIdx];
            end else begin
                sobolSeq <= sobolSeq;
            end
        end
    end

    /* initialization of directional vectors for current dimension*/
    `ifdef INWD3
        always_comb begin : proc_dirVec_3
            dirVec[0] <= 'd4;
            dirVec[1] <= 'd2;
            dirVec[2] <= 'd1;
        end

        always_comb begin : proc_3
            case(outoh)
                'b001 : vecIdx = 'd0;
                'b010 : vecIdx = 'd1;
                'b100 : vecIdx = 'd2;
                default : vecIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD4
        always_comb begin : proc_dirVec_4
            dirVec[0] <= 'd8;
            dirVec[1] <= 'd4;
            dirVec[2] <= 'd2;
            dirVec[3] <= 'd1;
        end

        always_comb begin : proc_4
            case(outoh)
                'b0001 : vecIdx = 'd0;
                'b0010 : vecIdx = 'd1;
                'b0100 : vecIdx = 'd2;
                'b1000 : vecIdx = 'd3;
                default : vecIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD5
        always_comb begin : proc_dirVec_5
            dirVec[0] <= 'd16;
            dirVec[1] <= 'd8;
            dirVec[2] <= 'd4;
            dirVec[3] <= 'd2;
            dirVec[4] <= 'd1;
        end

        always_comb begin : proc_5
            case(outoh)
                'b00001 : vecIdx = 'd0;
                'b00010 : vecIdx = 'd1;
                'b00100 : vecIdx = 'd2;
                'b01000 : vecIdx = 'd3;
                'b10000 : vecIdx = 'd4;
                default : vecIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD6
        always_comb begin : proc_dirVec_6
            dirVec[0] <= 'd32;
            dirVec[1] <= 'd16;
            dirVec[2] <= 'd8;
            dirVec[3] <= 'd4;
            dirVec[4] <= 'd2;
            dirVec[5] <= 'd1;
        end

        always_comb begin : proc_6
            case(outoh)
                'b000001 : vecIdx = 'd0;
                'b000010 : vecIdx = 'd1;
                'b000100 : vecIdx = 'd2;
                'b001000 : vecIdx = 'd3;
                'b010000 : vecIdx = 'd4;
                'b100000 : vecIdx = 'd5;
                default : vecIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD7
        always_comb begin : proc_dirVec_7
            dirVec[0] <= 'd64;
            dirVec[1] <= 'd32;
            dirVec[2] <= 'd16;
            dirVec[3] <= 'd8;
            dirVec[4] <= 'd4;
            dirVec[5] <= 'd2;
            dirVec[6] <= 'd1;
        end

        always_comb begin : proc_7
            case(outoh)
                'b0000001 : vecIdx = 'd0;
                'b0000010 : vecIdx = 'd1;
                'b0000100 : vecIdx = 'd2;
                'b0001000 : vecIdx = 'd3;
                'b0010000 : vecIdx = 'd4;
                'b0100000 : vecIdx = 'd5;
                'b1000000 : vecIdx = 'd6;
                default : vecIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD8
        always_comb begin : proc_dirVec_8
            dirVec[0] <= 'd128;
            dirVec[1] <= 'd64;
            dirVec[2] <= 'd32;
            dirVec[3] <= 'd16;
            dirVec[4] <= 'd8;
            dirVec[5] <= 'd4;
            dirVec[6] <= 'd2;
            dirVec[7] <= 'd1;
        end

        always_comb begin : proc_8
            case(outoh)
                'b00000001 : vecIdx = 'd0;
                'b00000010 : vecIdx = 'd1;
                'b00000100 : vecIdx = 'd2;
                'b00001000 : vecIdx = 'd3;
                'b00010000 : vecIdx = 'd4;
                'b00100000 : vecIdx = 'd5;
                'b01000000 : vecIdx = 'd6;
                'b10000000 : vecIdx = 'd7;
                default : vecIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD9
        always_comb begin : proc_dirVec_9
            dirVec[0] <= 'd256;
            dirVec[1] <= 'd128;
            dirVec[2] <= 'd64;
            dirVec[3] <= 'd32;
            dirVec[4] <= 'd16;
            dirVec[5] <= 'd8;
            dirVec[6] <= 'd4;
            dirVec[7] <= 'd2;
            dirVec[8] <= 'd1;
        end

        always_comb begin : proc_9
            case(outoh)
                'b000000001 : vecIdx = 'd0;
                'b000000010 : vecIdx = 'd1;
                'b000000100 : vecIdx = 'd2;
                'b000001000 : vecIdx = 'd3;
                'b000010000 : vecIdx = 'd4;
                'b000100000 : vecIdx = 'd5;
                'b001000000 : vecIdx = 'd6;
                'b010000000 : vecIdx = 'd7;
                'b100000000 : vecIdx = 'd8;
                default : vecIdx = 'd0;
            endcase // onehot
        end
    `endif

    `ifdef INWD10
        always_comb begin : proc_dirVec_10
            dirVec[0] <= 'd512;
            dirVec[1] <= 'd256;
            dirVec[2] <= 'd128;
            dirVec[3] <= 'd64;
            dirVec[4] <= 'd32;
            dirVec[5] <= 'd16;
            dirVec[6] <= 'd8;
            dirVec[7] <= 'd4;
            dirVec[8] <= 'd2;
            dirVec[9] <= 'd1;
        end

        always_comb begin : proc_10
            case(outoh)
                'b0000000001 : vecIdx = 'd0;
                'b0000000010 : vecIdx = 'd1;
                'b0000000100 : vecIdx = 'd2;
                'b0000001000 : vecIdx = 'd3;
                'b0000010000 : vecIdx = 'd4;
                'b0000100000 : vecIdx = 'd5;
                'b0001000000 : vecIdx = 'd6;
                'b0010000000 : vecIdx = 'd7;
                'b0100000000 : vecIdx = 'd8;
                'b1000000000 : vecIdx = 'd9;
                default : vecIdx = 'd0;
            endcase // onehot
        end
    `endif

endmodule