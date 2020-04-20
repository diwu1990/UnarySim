module SkewedSync # (
    parameter DEP=2
) (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input in0,
    input in1,
    output logic out0,
    output logic out1
);

    logic [DEP-1:0] cnt;

    logic cntFull;
    logic cntEmpty;
    logic in_ne;

    assign cntFull = &cnt;
    assign cntEmpty = ~|cnt;
    assign in_ne = (in0 ^ in1);

    always_ff @(posedge clk or negedge rst_n) begin : proc_cnt
        if(~rst_n) begin
            cnt <= 0;
        end else begin
            if (in_ne) begin
                /*
                if(in0) begin
                    cnt <= cntFull ? cnt : (cnt + 1);
                end else begin
                    cnt <= cntEmpty ? cnt : (cnt - 1);
                end
                */
                cnt <= (in0 ? cntFull : cntEmpty) ? cnt : (cnt + (in0 ? 1 : -1));
            end else begin
                cnt <= cnt;
            end
        end
    end

    always_comb begin : proc_out0
        if (in_ne) begin
            if(in0) begin
                out0 <= cntFull ? 1 : 0;
            end else begin
                out0 <= cntEmpty ? 0 : 1;
            end
        end else begin
            out0 <= in0;
        end
    end

    assign out1 = in1;

endmodule