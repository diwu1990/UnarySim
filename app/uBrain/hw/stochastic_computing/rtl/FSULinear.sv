`include "FSUAdd.sv"
`include "BufferLoadArray.sv"
`include "Mul.sv"
`include "SobolRngDim1.sv"

module FSULinear #(
    parameter IDIM = 8,
    parameter IWID = 10,
    parameter ODIM = 2,
    parameter OWID = IWID,
    parameter RWID = IWID,
    parameter SDIM = 32,
    parameter BDEP = 999
) (
    input logic clk,
    input logic rst_n,
    input logic load,
    input logic sel,
    input logic clear,
    input logic iFbit [IDIM - 1 : 0],
    input logic iFb_n [IDIM - 1 : 0],
    input logic [IWID - 1 : 0] iWeig [ODIM * IDIM - 1 : 0],
    output logic oFbit [ODIM - 1 : 0]
);

    // input: unary
    // output: unary
    // weight: binary

    logic [IWID - 1 : 0] oWeig [ODIM * IDIM - 1 : 0];
    logic [IWID - 1 : 0] tWeig [ODIM * IDIM - 1 : 0];
    // rng are shared by weights with identical inputs
    logic [RWID - 1 : 0] wRng [IDIM - 1 : 0];
    logic [RWID - 1 : 0] wR_n [IDIM - 1 : 0];
    logic oFmbs [ODIM * IDIM - 1 : 0];

    // load weight
    BufferLoadArray #(
        .IWID(IWID),
        .IDIM(ODIM * IDIM)
    ) U_BufferLoadArray_iWeig (
        .clk(clk),
        .rst_n(rst_n),
        .load(load),
        .iData(iWeig),
        .oData(oWeig)
    );

    genvar i, j;
    generate
        // generate weight rngs
        for (i = 0; i < IDIM; i = i + 1) begin : gen_weight_rng
            SobolRngDim1 #(
                .RWID(RWID)
            ) U_SobolRngDim1_weig_rng(
                .clk(clk),
                .rst_n(rst_n),
                .enable(iFbit[i]),
                .sobolSeq(wRng[i])
                );
            SobolRngDim1 #(
                .RWID(RWID)
            ) U_SobolRngDim1_we_n_rng(
                .clk(clk),
                .rst_n(rst_n),
                .enable(iFb_n[i]),
                .sobolSeq(wR_n[i])
                );
        end

        // generate xnor multiplier, one of the inverted input is already generated
        for (i = 0; i < ODIM; i = i + 1) begin : gen_oFmap_prod
            for (j = 0; j < IDIM; j = j + 1) begin
                Mul # (
                    .IWID(IWID)
                ) U_Mul(
                    .iDbit(iFbit[j]),
                    .iDb_n(iFb_n[j]),
                    .iWeig(oWeig[i * IDIM + j]),
                    .iWRng(wRng[j]),
                    .iWR_n(wR_n[j]),
                    .oDbit(oFmbs[i * IDIM + j])
                );
            end
        end

        // generate fsu add
        for (i = 0; i < ODIM; i = i + 1) begin : gen_output_bs
            FSUAdd #(
                .IDIM(IDIM),
                .IWID(IWID),
                .SCAL(1),
                .OFST(8),
                .BDEP(BDEP)
            ) U_FSUAdd_bs(
                .clk(clk),
                .rst_n(rst_n),
                .iBit(oFmbs[(i + 1) * IDIM - 1 : i * IDIM]),
                .oBit(oFbit)
            );
        end
    endgenerate
    
endmodule