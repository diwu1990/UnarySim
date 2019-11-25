module add16 (
    input [3:0] randNum,
    input [15:0] in,
    output out
);

    always_comb begin : proc_
        case (randNum)
            4'h0: out = in[0];
            4'h1: out = in[1];
            4'h2: out = in[2];
            4'h3: out = in[3];
            4'h4: out = in[4];
            4'h5: out = in[5];
            4'h6: out = in[6];
            4'h7: out = in[7];
            4'h8: out = in[8];
            4'h9: out = in[9];
            4'ha: out = in[10];
            4'hb: out = in[11];
            4'hc: out = in[12];
            4'hd: out = in[13];
            4'he: out = in[14];
            4'hf: out = in[15];
            default : out = 0;
        endcase
    end

endmodule