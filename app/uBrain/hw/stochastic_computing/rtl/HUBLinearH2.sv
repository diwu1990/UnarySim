`include "AdderTree.sv"
`include "BufferDoubleArray.sv"
`include "BufferLoadArray.sv"
`include "Cmp.sv"
`include "HActArray.sv"
`include "Mul.sv"
`include "SobolRngDim1.sv"

module HUBLinearF4 #(
    parameter IDIM = 8,
    parameter IWID = 10,
    parameter ODIM = 2,
    parameter OWID = IWID,
    parameter RWID = IWID,
    parameter RELU = 1,
    parameter SDIM = 32,
    parameter FOLD = 4,
    parameter BDEP = 999,
    parameter BDIM = 2 ** ($clog2(IDIM) - $clog2(SDIM)),
    parameter TDIM = (BDIM < 1) ? 1 : BDIM
) (
    input logic clk,
    input logic rst_n,
    input logic load,
    input logic sel,
    input logic clear,
    input logic [1 : 0] part,
    input logic [IWID - 1 : 0] iFmap [IDIM - 1 : 0],
    input logic [IWID - 1 : 0] iWeig [ODIM * IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oFmap [ODIM - 1 : 0]
);

    logic [RWID - 1 : 0] iRng [TDIM * SDIM - 1 : 0];
    logic iBit [IDIM - 1 : 0];
    logic iB_n [IDIM - 1 : 0];
    logic [IWID - 1 : 0] oWeig [ODIM * IDIM - 1 : 0];
    logic [IWID - 1 : 0] tWeig [ODIM / FOLD * IDIM - 1 : 0];
    // rng are shared by weights with identical inputs
    logic [RWID - 1 : 0] wRng [IDIM - 1 : 0];
    logic [RWID - 1 : 0] wR_n [IDIM - 1 : 0];
    logic oFmbs [ODIM / FOLD * IDIM - 1 : 0];
    logic [$clog2(IDIM) + 1 - 1 : 0] oFbin0 [ODIM - 1 : 0];
    logic [$clog2(IDIM) + 1 + OWID - 1 : 0] oFbin1 [ODIM - 1 : 0];

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

    RngShareArray #(
        .RWID(RWID),
        .BDIM(BDIM),
        .SDIM(SDIM)
    ) U_RngShareArray(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .rngSeq(iRng)
        );

    genvar i, j;
    generate
        // generate input bitstreams
        for (i = 0; i < IDIM; i = i + 1) begin : gen_iFmap_bs
            Cmp # (
                .IWID(IWID)
            ) U_Cmp_iFmap(
                .iData(iFmap[i]),
                .iRng(iRng[i]),
                .oBit(iBit[i])
                );
            assign iB_n[i] = ~iBit[i];
        end

        // generate weight rngs
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

        // generate xnor multiplier, one of the inverted input is already generated
        always_comb begin
            case (part)
                0 : tWeig = oWeig[ODIM / FOLD * IDIM * 1 - 1 : ODIM / FOLD * IDIM * 0];
                1 : tWeig = oWeig[ODIM / FOLD * IDIM * 2 - 1 : ODIM / FOLD * IDIM * 1];
                2 : tWeig = oWeig[ODIM / FOLD * IDIM * 3 - 1 : ODIM / FOLD * IDIM * 2];
                3 : tWeig = oWeig[ODIM / FOLD * IDIM * 4 - 1 : ODIM / FOLD * IDIM * 3];
            endcase
        end
        for (i = 0; i < ODIM / FOLD; i = i + 1) begin : gen_oFmap_bs
            for (j = 0; j < IDIM; j = j + 1) begin
                Mul # (
                    .IWID(IWID)
                ) U_Mul(
                    .iDbit(iBit[j]),
                    .iDb_n(iB_n[j]),
                    .iWeig(tWeig[i * IDIM + j]),
                    .iWRng(wRng[j]),
                    .iWR_n(wR_n[j]),
                    .oDbit(oFmbs[i * IDIM + j])
                );
            end
        end

        // generate parallel counter
        for (i = 0; i < ODIM; i = i + 1) begin : gen_oFmap_bin
            AdderTree #(
                .IDIM(IDIM),
                .IWID(1),
                .BDEP(BDEP)
            ) U_AdderTree_oFmap_bin(
                .clk(clk),
                .rst_n(rst_n),
                .iData(oFmbs[(i + 1) * IDIM - 1 : i * IDIM]),
                .oData(oFbin0[i])
            );
        end

        for (i = 0; i < FOLD; i = i + 1) begin
            BufferDoubleArray #(
                .IDIM(ODIM),
                .IWID($clog2(IDIM) + 1),
                .OWID($clog2(IDIM) + 1 + OWID)
            ) U_BufferDoubleArray(
                .clk(clk),
                .rst_n(rst_n),
                .iAccSel(sel),
                .iClear(clear | ~(part == i)),
                .iData(oFbin0),
                .oData(oFbin1[ODIM / FOLD * (i + 1) - 1 : ODIM / FOLD * i])
            );
        end

        HActArray #(
            .IDIM(ODIM),
            .IWID($clog2(IDIM) + 1 + OWID),
            .ADIM(IDIM),
            .OWID(OWID),
            .RELU(RELU)
        ) U_HActArray(
            .clk(clk),
            .rst_n(rst_n),
            .iData(oFbin1),
            .oData(oFmap)
        );
    endgenerate
    
endmodule