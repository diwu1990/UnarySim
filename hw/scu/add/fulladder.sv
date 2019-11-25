module fulladder (
    input a,
    input b,
    input ci,
    output sum,
    output co
);

    assign {co,sum} = a + b + ci;

endmodule