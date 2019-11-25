`timescale 1ns/1ns
`include "SobolRNGDef.sv"

module LSZ_tb ();

    logic [`INWD-1:0]in;
    logic [`LOGINWD-1:0]out;

    LSZ U_LSZ(
        .in(in),
        .lszidx(out)
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