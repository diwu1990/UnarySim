`include "FSUAdd.sv"
`include "BufferLoadArray.sv"
`include "Mul.sv"
`include "SobolRngDim1.sv"

module FSULinear #(
    parameter IDIM = 8,
    parameter IWID = 10,
    parameter ODIM = 2,
    parameter OWID = IWID,
    parameter CWID = IWID,
    parameter SDIM = 32,
    parameter BDEP = 999,
    parameter BDIO = 2 ** ($clog2(ODIM) - $clog2(SDIM)),
    parameter TDIO = (BDIO < 1) ? 1 : BDIO
) (
    input logic clk,
    input logic rst_n,
    input logic load,
    input logic sel,
    input logic clear,
    input logic iFbit [IDIM - 1 : 0],
    input logic [IWID - 1 : 0] iWeig [ODIM * IDIM - 1 : 0],
    output logic oFbit [ODIM - 1 : 0]
);

    // input: unary
    // output: unary
    // weight: binary

    logic [CWID - 1 : 0] wCnt [TDIO * SDIM - 1 : 0];
    logic [IWID - 1 : 0] oWeig [ODIM * IDIM - 1 : 0];
    logic [IWID - 1 : 0] tWeig [ODIM * IDIM - 1 : 0];
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

    CntShareArray #(
        .CWID(CWID),
        .BDIM(BDIO),
        .SDIM(SDIM)
    ) U_CntShareArray_weigh_rng(
        .clk(clk),
        .rst_n(rst_n),
        .enable(1'b1),
        .rngSeq(wCnt)
        );

    genvar i, j;
    generate
        // generate weight cnt
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