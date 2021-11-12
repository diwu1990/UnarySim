`timescale 1ns/1ns
`include "../BufferLoadArray.sv"

module BufferLoadArray_tb();
    parameter IWID = 8;
    parameter IDIM = 4;

    logic clk;
    logic rst_n;
    logic load;
    logic [IWID - 1 : 0] iData [IDIM - 1 : 0];
    logic [IWID - 1 : 0] oData [IDIM - 1 : 0];

    BufferLoadArray U_BufferLoadArray(
        .clk(clk),
        .rst_n(rst_n),
        .load(load),
        .iData(iData),
        .oData(oData)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("BufferLoadArray.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif


    initial
    begin
        clk = 1;
        rst_n = 0;
        load = 1;
        
        #15;
        rst_n = 1;
        iData = {'d10, 'd1, 'd8, 'd9};

        #10;
        iData = {'d10, 'd2, 'd8, 'd9};

        #10;
        load = 0;

        #10;
        iData = {'d10, 'd2, 'd8, 'd9};

        #400;
        $finish;
    end

    
endmodule