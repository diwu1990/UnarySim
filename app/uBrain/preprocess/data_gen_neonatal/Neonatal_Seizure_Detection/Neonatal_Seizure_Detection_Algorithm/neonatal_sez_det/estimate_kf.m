function kf = estimate_kf(A);    
    
B = size(A);
    N = B(1); n = B(2); k = 2;
    eta = [sum(A')' n-sum(A')'];
    p = 1/N/n * sum(eta);
    P = 1./(n*(n-1)).*(sum(eta'.^2)-n);
    Pbar = 1/N*(sum(P));
    Pebar = sum(p.^2);
    kf = (Pbar-Pebar)/(1-Pebar);
