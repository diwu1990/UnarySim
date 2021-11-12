`include "AdderTree.sv"
`include "BufferDoubleArray.sv"
`include "BufferLoadArray.sv"
`include "Cmp.sv"
`include "HActArray.sv"
`include "Mul.sv"
`include "SobolRngDim1.sv"

module HUBLinearH0 #(
    parameter IDIM = 64,
    parameter IWID = 10,
    parameter ODIM = 2,
    parameter OWID = IWID,
    parameter RWID = IWID,
    parameter RELU = 1,
    parameter BDEP = 999
) (
    input logic clk,
    input logic rst_n,
    input logic load,
    input logic sel,
    input logic clear,
    input logic [IWID - 1 : 0] iFmap [IDIM - 1 : 0],
    input logic [IWID - 1 : 0] iWeig [ODIM * IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oFmap [ODIM - 1 : 0]
);

    logic [IWID - 1 : 0] iRng [IDIM - 1 : 0];
    logic iBit [IDIM - 1 : 0];
    logic iB_n [IDIM - 1 : 0];
    logic [IWID - 1 : 0] oWeig [ODIM * IDIM - 1 : 0];
    // rng are shared by weights with identical inputs
    logic [IWID - 1 : 0] wRng [IDIM - 1 : 0];
    logic [IWID - 1 : 0] wR_n [IDIM - 1 : 0];
    logic [IWID - 1 : 0] oFmbs [ODIM * IDIM - 1 : 0];
    logic [$clog2(IDIM) + 1 - 1 : 0] oFmbin [ODIM - 1 : 0];

    BufferLoad #(
        .IWID(IWID),
        .IDIM(IDIM)
    ) U_BufferLoad (
        .clk(clk),
        .rst_n(rst_n),
        .load(load),
        .iData(iWeig),
        .oData(oWeig)
    );

    genvar i, j;
    generate
        for (i = 0; i < IDIM; i = i + 1) begin : gen_iFmap_bs
            SobolRngDim1 #(
                .RWID(RWID)
            ) U_SobolRngDim1_iFmap(
                .clk(clk),
                .rst_n(rst_n),
                .enable('b1),
                .sobolSeq(iRng)
                );
            Cmp # (
                .IWID(IWID)
            ) U_Cmp_iFmap(
                .iData(iFmap[i]),
                .iRng(iRng[i]),
                .oBit(iBit[i])
                );
            assign iB_n[i] = ~iBit[i];
        end

        for (i = 0; i < IDIM; i = i + 1) begin : gen_weight_rng
            SobolRngDim1 #(
                .RWID(RWID)
            ) U_SobolRngDim1_weig_rng(
                .clk(clk),
                .rst_n(rst_n),
                .enable(iBit[i]),
                .sobolSeq(wRng[i])
                );
            SobolRngDim1 #(
                .RWID(RWID)
            ) U_SobolRngDim1_we_n_rng(
                .clk(clk),
                .rst_n(rst_n),
                .enable(iB_n[i]),
                .sobolSeq(wR_n[i])
                );
        end

        for (i = 0; i < ODIM; i = i + 1) begin : gen_oFmap_bs
            for (j = 0; j < IDIM; j = j + 1) begin
                Mul # (
                    .IWID(IWID)
                ) U_Mul(
                    .iDbit(iBit[j]),
                    .iDb_n(iB_n[j]),
                    .iWeig(oWeig[i * IDIM + j]),
                    .iWRng(wRng[j]),
                    .iWR_n(wR_n[j]),
                    .oDbit(oFmbs[i * IDIM + j])
                );
            end
        end

        for (i = 0; i < ODIM; i = i + 1) begin : gen_oFmap_bin
            AdderTree #(
                .IDIM(IDIM),
                .IWID(1),
                .BDEP(BDEP)
            ) U_AdderTree_oFmap_bin(
                .clk(clk),
                .rst_n(rst_n),
                .iData(oFmbs[(i + 1) * IDIM - 1 : i * IDIM]),
                .oData(oFmbin[i])
            );
        end

        HActArray #(
            .IDIM(ODIM),
            .IWID($clog2(IDIM) + 1),
            .ADIM(IDIM),
            .OWID(OWID),
            .RELU(RELU)
        ) U_HActArray(
            .clk(clk),
            .rst_n(rst_n),
            .iData(oFmbin),
            .oData(oFmap)
        );
    endgenerate
    
endmodule