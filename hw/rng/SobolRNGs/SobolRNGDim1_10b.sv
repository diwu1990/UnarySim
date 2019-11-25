module SobolRNGDim1_10b (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic enable,
    output logic sobolSeq
);
    
    parameter INWD = 10;
    parameter LOGINWD = 4;

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

endmodule