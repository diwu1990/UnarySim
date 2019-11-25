`timescale 1ns/1ns
`include "SobolSeq2Def.sv"

module LSZ2_tb ();

    logic [`INWIDTH-1:0]in;
    logic [`LOGINWIDTH-1:0]out;

    LSZ2 U_LSZ2(
        .in(in),
        .out(out)
        );

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("LSZ2.fsdb");
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