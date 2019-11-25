module tMUL_uni (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [7:0] iA,
    input logic [7:0] iB,
    input logic loadA,
    input logic loadB,
    input logic [7:0] sobolSeq,
    output logic oC,
    output logic stop
);
    
    logic [7:0] iA_buf;
    logic [7:0] iB_buf;
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_iA_buf
        if(~rst_n) begin
            iA_buf <= 0;
            stop <= 1;
        end else begin
            if(loadA) begin
                iA_buf <= iA;
                stop <= 1;
            end else begin
                if(iA_buf != 0) begin
                    iA_buf <= iA_buf - 1;
                    stop <= 0;
                end else begin
                    iA_buf <= iA_buf;
                    stop <= 1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : proc_iB_buf
        if(~rst_n) begin
            iB_buf <= 0;
        end else begin
            if(loadB) begin
                iB_buf <= iB;
            end else begin
                iB_buf <= iB_buf;
            end
        end
    end

    always_comb begin : proc_oC
        oC <= ~stop & (iB_buf > sobolSeq);
    end

endmodule