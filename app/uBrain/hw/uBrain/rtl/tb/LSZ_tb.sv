`timescale 1ns/1ns
`include "../LSZ.sv"

module LSZ_tb ();

    parameter IWID = 4;
    parameter IWL2 = $clog2(IWID);

    logic [IWID - 1:0]in;
    logic [IWL2 - 1:0]out;

    LSZ # (
        .IWID(IWID)
    ) U_LSZ(
        .in(in),
        .lszIdx(out)
        );

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("LSZ.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif

    initial
    begin
        in = 0;
        
        #15;
        repeat(500) begin
            #10 in = in+1;
        end
        $finish;
    end

endmodule