module add2 (
    input randNum,
    input [1:0] in,
    output out
);

    assign out = randNum ? in[1] : in[0];

endmodule
