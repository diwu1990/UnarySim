`include "AdderTree.sv"
`include "BufferDoubleArray.sv"
`include "BufferLoadArray.sv"
`include "Cmp.sv"
`include "HActArray.sv"
`include "Mul.sv"
`include "MuxArray.sv"
`include "SobolRngDim1.sv"
`include "RngShareArray.sv"

module HUBLinearFold #(
    parameter IDIM = 8,
    parameter IWID = 10,
    parameter ODIM = 2,
    parameter OWID = IWID,
    parameter RWID = IWID,
    parameter CWID = IWID,
    parameter RELU = 1,
    parameter SDIM = 32,
    parameter FOLD = 1,
    parameter PWID = ($clog2(FOLD) < 2) ? 1 : $clog2(FOLD),
    parameter BDEP = 999,
    parameter BDII = 2 ** ($clog2(IDIM) - $clog2(SDIM)),
    parameter TDII = (BDII < 1) ? 1 : BDII,
    parameter BDIO = 2 ** ($clog2(ODIM) - $clog2(SDIM)) / FOLD,
    parameter TDIO = (BDIO < 1) ? 1 : BDIO
) (
    input logic clk,
    input logic rst_n,
    input logic load,
    input logic sel,
    input logic clear,
    input logic [PWID - 1 : 0] part,
    input logic [IWID - 1 : 0] iFmap [IDIM - 1 : 0],
    input logic [IWID - 1 : 0] iWeig [ODIM * IDIM - 1 : 0],
    output logic [OWID - 1 : 0] oFmap [ODIM - 1 : 0]
);

    // input: binary
    // output: binary
    // weight: binary
    
    logic [RWID - 1 : 0] iRng [TDII * SDIM - 1 : 0];
    logic [CWID - 1 : 0] wCnt [TDIO * SDIM - 1 : 0];
    logic iBit [IDIM - 1 : 0];
    logic [IWID - 1 : 0] oWeig [ODIM * IDIM - 1 : 0];
    logic [IWID - 1 : 0] tWeig [ODIM / FOLD * IDIM - 1 : 0];
    logic wBit [ODIM / FOLD * IDIM - 1 : 0];
    logic oFmbs [ODIM / FOLD * IDIM - 1 : 0];
    logic [$clog2(IDIM) + 1 - 1 : 0] oFbin0 [ODIM / FOLD - 1 : 0];
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
        .BDIM(BDII),
        .SDIM(SDIM)
    ) U_RngShareArray_iFmap_rng(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .rngSeq(iRng)
        );
    
    CntShareArray #(
        .CWID(CWID),
        .BDIM(BDIO),
        .SDIM(SDIM)
    ) U_CntShareArray_weigh_rng(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .cntSeq(wCnt)
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
        end

        // select the correct weight for accumulation
        MuxArray #(
            .IDIM(ODIM * IDIM),
            .IWID(IWID),
            .FOLD(FOLD)
        ) U_MuxArray (
            .iData(oWeig),
            .iSel(part),
            .oData(tWeig)
        );

        // generate xnor multiplier, one of the inverted input is already generated
        for (i = 0; i < ODIM / FOLD; i = i + 1) begin : gen_oFmap_bs
            for (j = 0; j < IDIM; j = j + 1) begin
                Cmp # (
                    .IWID(IWID)
                ) U_Cmp_iFmap(
                    .iData(tWeig[i * IDIM + j]),
                    .iRng(wCnt[i * IDIM + j]),
                    .oBit(wBit[i * IDIM + j])
                    );
                assign oFmbs[i * IDIM + j] = ~(wBit[i * IDIM + j] ^ iBit[j]);
            end
        end

        // generate parallel counter
        for (i = 0; i < ODIM / FOLD; i = i + 1) begin : gen_oFmap_bin
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

        for (i = 0; i < FOLD; i = i + 1) begin : gen_weight_double_buffer
            BufferDoubleArray #(
                .IDIM(ODIM / FOLD),
                .IWID($clog2(IDIM) + 1),
                .OWID($clog2(IDIM) + 1 + OWID)
            ) U_BufferDoubleArray(
                .clk(clk),
                .rst_n(rst_n),
                .iAccSel(sel),
                .iClear(clear),
                .iHold(~(part == i)),
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