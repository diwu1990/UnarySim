`include "param_def.sv"

module data_buf (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic rd_en,
    input logic rd_addr,
    input logic wr_en,
    input logic wr_addr,
    input logic [`MAC_BW-1 : 0] iData,
    output logic [`MAC_BW-1 : 0] oData
);

    // each read/write deals with one row of data
    logic [`MAC_BW-1 : 0][`ROW_CNT-1 : 0] dataMem;

    always_ff @(posedge clk or negedge rst_n) begin : proc_dataMem
        if(~rst_n) begin
            dataMem <= 0;
        end else begin
            if(wr_en) begin
                dataMem[wr_addr] <= iData;
            end
        end
    end

    always_comb begin : proc_rd_bus
        if (rd_en) begin
            oData <= dataMem[rd_addr];
        end else begin
            oData <= oData;
        end
    end

endmodule