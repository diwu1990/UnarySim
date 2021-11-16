`timescale 1ns/1ns
`include "../MuxArray.sv"

module MuxArray_tb ();

    parameter IDIM = 8;
    parameter IWID = 8;
    parameter FOLD = 4;
    parameter SWID = ($clog2(FOLD) < 1) ? 1 : $clog2(FOLD);
    parameter ODIM = IDIM / FOLD;
    parameter OWID = IWID;

    logic clk;
    logic rst_n;
    logic [IWID - 1 : 0] iData [IDIM - 1 : 0];
    logic [SWID - 1 : 0] iSel;
    logic [OWID - 1 : 0] oData [ODIM - 1 : 0];

    MuxArray # (
        .IDIM(IDIM),
        .IWID(IWID),
        .FOLD(FOLD)
    ) U_MuxArray(
        .iData(iData),
        .iSel(iSel),
        .oData(oData)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("MuxArray.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif

    initial
    begin
        iData = {'d32, 'd64, 'd96, 'd128, 'd160, 'd192, 'd224, 'd255};
        iSel = 'd0;

        #20;
        iSel = 'd1;

        #20;
        iSel = 'd2;

        #20;
        iSel = 'd3;

        #20;
        iSel = 'd4;

        #20;
        iSel = 'd5;

        #20;
        iSel = 'd6;

        #20;
        iSel = 'd7;
        
        #100;
        $finish;
    end

endmodule