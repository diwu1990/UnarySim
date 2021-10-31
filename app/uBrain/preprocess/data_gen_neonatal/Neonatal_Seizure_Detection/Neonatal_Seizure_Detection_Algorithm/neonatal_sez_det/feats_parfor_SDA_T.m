function feats=feats_parfor_SDA_T(n,sig,sc,len,fs_orig,fs,Num,Den,olap)
pp=gcp;
if pp.Connected~=1
    parpool(n)
end
D = parallel.pool.DataQueue;
h=waitbar(0, ['Computing features with ',num2str(n),' parallel pools']);
afterEach(D, @nUpdateWaitbar);
p=1;N=length(sig);
feats=cell(N,1);
parfor i=1:N
    feats{i}=ch_temko(sig,sc,len, fs_orig,fs,Num,Den,olap,i);
    send(D,i);
end
function nUpdateWaitbar(~)
    waitbar(p/N, h,['Computed features on channels ',num2str(p),'/',num2str(N)]);
    p = p + 1;
end
close(h)

end