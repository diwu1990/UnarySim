`timescale 1ns/1ns
`include "../HUBLinearH0.sv"

module HUBLinearH0_tb ();
    parameter IDIM = 4;
    parameter IWID = 8;
    parameter ODIM = 2;
    parameter RELU = 1;
    parameter BDEP = 999;
    parameter OWID = IWID;
    parameter RWID = IWID;

    logic clk;
    logic rst_n;
    logic load;
    logic sel;
    logic clear;
    logic [IWID - 1 : 0] iFmap [IDIM - 1 : 0];
    logic [IWID - 1 : 0] iWeig [ODIM * IDIM - 1 : 0];
    logic [OWID - 1 : 0] oFmap [ODIM - 1 : 0];

    HUBLinearH0 # (
        .IDIM(IDIM),
        .IWID(IWID),
        .ODIM(ODIM),
        .RELU(RELU),
        .BDEP(BDEP)
    ) U_HUBLinearH0(
        .clk(clk),    // Clock
        .rst_n(rst_n),  // Asynchronous reset active low
        .load(load),
        .sel(sel),
        .clear(clear),
        .iFmap(iFmap),
        .iWeig(iWeig),
        .oFmap(oFmap)
        );

    // clk define
    always #5 clk = ~clk;

    `ifdef DUMPFSDB
        initial begin
            $fsdbDumpfile("HUBLinearH0.fsdb");
            $fsdbDumpvars(0,"+all");
            // $fsdbDumpvars;
        end
    `endif

    initial
    begin
        clk = 1;
        rst_n = 0;
        load = 0;
        sel = 0;
        clear = 0;
        iFmap = {'d31, 'd63, 'd95, 'd127};
        iWeig = {'d31, 'd63, 'd95, 'd127, 'd159, 'd191, 'd224, 'd255};

        #100;
        rst_n = 1;
        load = 1;
        clear = 1;

        #10
        load = 0;
        clear = 0;

        #(10*(2**IWID))
        sel = 1;
        clear = 1;
        iFmap = {'d31, 'd63, 'd95, 'd127};
        iWeig = {'d159, 'd191, 'd224, 'd255, 'd31, 'd63, 'd95, 'd127};
        
        #10
        clear = 1;
        
        #(10*(2**IWID))
        sel = 0;

        #100;
        $finish;
    end

endmodule