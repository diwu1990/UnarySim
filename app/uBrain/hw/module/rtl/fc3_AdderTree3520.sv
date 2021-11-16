module fc3_AdderTree3520 #(
    parameter IDIM = 3520, // input dimension
    parameter IDL2 = $clog2(IDIM), // log2(IDIM)
    parameter IWID = 10, // input width
    parameter OWID = IWID + IDL2, // output width
    parameter BDEP = 2 // depth to insert buffers for the pipeline
) (
    input logic clk,
    input logic rst_n,
    input logic [IWID - 1 : 0] iData [IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oData
);

    // define temp data used in the adder tree
    // each adder has two inputs
    logic [OWID - 1 : 0] tData [(2 * (2**IDL2) - 1) - 1 : 0];

    genvar i, j;
    // packing input to output width
    generate
        for (i = 0; i < 2**IDL2; i = i + 1) begin : initialloop
            if (i < IDIM) begin : assigninput
                assign tData[i] = iData[i];
            end
            else begin : assignzero
                assign tData[i] = 'b0;
            end
        end
    endgenerate

    // build adders at each pipeline
    generate
        for (i = 0; i < IDL2; i = i + 1) begin : firstloop
            for (j = 0; j < 2**(IDL2 - i - 1); j = j + 1) begin : secondloop
                if ((i + 1) % BDEP == 0) begin : sequential
                    // insert buffers
                    always @(posedge clk or negedge rst_n) begin
                        if (~rst_n) begin
                            tData[(2 * (2**IDL2) - 2**(IDL2 - i)) + j] <= 'b0;
                        end
                        else begin
                            tData[(2 * (2**IDL2) - 2**(IDL2 - i)) + j] <= tData[(2 * (2**IDL2) - 2**(IDL2 - i)) - 2**(IDL2 - i) + 2 * j] + 
                                                                                tData[(2 * (2**IDL2) - 2**(IDL2 - i)) - 2**(IDL2 - i) + 2 * j + 1];
                        end
                    end
                end
                else begin : combinational
                    // insert no buffers
                    always_comb begin
                        tData[(2 * (2**IDL2) - 2**(IDL2 - i)) + j] <= tData[(2 * (2**IDL2) - 2**(IDL2 - i)) - 2**(IDL2 - i) + 2 * j] + 
                                                                            tData[(2 * (2**IDL2) - 2**(IDL2 - i)) - 2**(IDL2 - i) + 2 * j + 1];
                    end
                end
            end
        end
    endgenerate

    assign oData = tData[(2 * (2**IDL2) - 1) - 1];
    
endmodule