function D2=d2descriptor(vertex,m,edges)
 
idx=randi(size(vertex,1),m,2);

v1all=vertex(idx(:,1),:);
v2all=vertex(idx(:,2),:);

X=sqrt(sum((v1all-v2all).^2,2));

D2=histcounts(X,edges);

end