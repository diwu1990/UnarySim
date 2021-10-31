 function r = freq_corr(tfd, M)

A = size(tfd);
p1 = zeros(A(1), A(1));
extent = A(2)/16; x = 1:A(2);
% Loop through time-slices
for z1 = 1:A(1);
    % time slices for correlation
    MM = z1+M; if MM>A(1); MM = A(1); end 
    % slice for comparison
    ts = tfd(z1,:);
    % Loop through following 3 slices for which correlation is computed
    for z2 = z1+1:MM
        % time slice to be scaled
          y = tfd(z2,:);  
          % Maximum of time-slice
          rf1 = find(y==max(y), 1);
          % limits for frequency samples
          ulim = round(round(10*(rf1+extent)/rf1)/10*A(2));
           llim = round(round(10*(rf1-extent)/rf1)/10*A(2));
           scls = [linspace(llim,A(2), extent+1) linspace(A(2), ulim, extent+1)];
           scls = unique(round(scls));
            scls = scls(scls>0);
            sc = zeros(length(scls),A(2)); p = zeros(1,length(scls));
            for ii = 1:length(scls);
                xx = linspace(1,A(2),scls(ii));
                yy = pchip(x, y, xx); % Hermite Polynomial Fit (cubic splines are too oscillatory)
                if scls(ii) > A(2); % expanded
                    yy = yy(1:A(2));    % Note: no normalisation for changes in frequency scale
                else % contracted
                    yy = [yy zeros(1, A(2)-length(yy))];
                end
                sc(ii,:) = yy./max(yy); % Normalisation 
                p(ii) = corr(yy', ts');
            end 
            % Highest correlation chosen from different scales
            p1(z1, z2) = max(p); 
    end
end

rr = [];
for ii = 1:M
    rr = [rr ; diag(p1, ii)];
end

r = [quantile(rr, [0.1 0.5 0.9]) mean(rr)];
