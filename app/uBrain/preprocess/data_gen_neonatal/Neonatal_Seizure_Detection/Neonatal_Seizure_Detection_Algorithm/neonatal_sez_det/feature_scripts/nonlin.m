function [N,ll] = nonlin(y_step,v)

ll = 0;
for i=1:v-1
    L = abs(diff(y_step(i:i+1)));
    ll =  ll + L;
end
N = nonlinear_energy(y_step);