function total_shapes=shape_clust_CVPR(foldername,num_exem,foldersave,foldersaveb)

[vall,listings]=read_off_files(foldername);
disp('off files read');


total_shapes=length(listings);


%% D2 descriptor
edges=linspace(0,120,200);
m=10^6;
d2all=cell(length(listings),1);
for kk=1:length(vall)
    vertex=vall{kk};
    D2=d2descriptor(vertex,m,edges)/m;
    d2all{kk}=D2;
    if mod(kk,100)==0
        disp(kk);
    end
end

d2all=cell2mat(d2all);
d2dist=pdist2(d2all,d2all);



%% cluster

s=-d2dist;




kkk=2*num_exem;

    
con=10;
sk=sparse(exp(con*s));
d=sparse(diag(sum(sk)));
L=d-sk;
dsym=sparse(diag(sqrt(1./sum(sk))));
Lsym=dsym*L*dsym;
[V,D]=eigs(Lsym,kkk+1,-1e-5);
[~,idx]=sort(diag(D));
V=V(:,idx);
V=V./(ones(size(V,1),1)*sqrt(sum(V.^2)));
% V=V./(sqrt(sum(V.^2,2))*ones(1,size(V,2)));
[idx,C] = kmeans(V(:,2:end),kkk);
ik=pdist2(V(:,2:end),C);
[~,idxx]=min(ik);

histidx=histc(idx,1:max(idx));
[~,sortidx]=sort(histidx,'descend');
% idxx(histidx<thresh)=[];
idxx=idxx(sortidx);
exemplars=listings(idxx(1:num_exem));

%% display the cluster centers


close all
% foldername='H:\ShapeNetCore.v1\ShapeNetCore.v1\02691156\Off_files\';
% foldersave='H:\ShapeNetCore.v1\ShapeNetCore.v1\02691156\Off_blm\';
figure;
for kk=1:length(exemplars)
    
    [vertex,face] = read_off([foldername,exemplars(kk).name]);
    vertex=vertex';
    subplot(1,length(exemplars),kk);
    trimesh(face',vertex(:,1),vertex(:,2),vertex(:,3),'facecolor','interp');
    axis off
    axis equal
end

x = input('These are the exemplars. Press enter');

if x==1
    a;
    disp('Error');
end

close all
%% copy off files and number them 1 to n

mkdir(foldersave);
for kk=1:length(listings)  
     [vertex,face] = read_off([foldername,listings(kk).name]);
     filename=[foldersave,num2str(kk),'-mesh.off'];
     write_off(filename, vertex, face);
    if mod(kk,100)==0
        disp(kk);
    end
end


%base-files
mkdir(foldersaveb);
for kk=1:length(exemplars)  
   [vertex,face] = read_off([foldername,exemplars(kk).name]);
    filename=[foldersaveb,num2str(kk),'-meshexem.off'];
    write_off(filename, vertex, face);
end





%base-files-self
for kk=1:length(exemplars)
    [vertex,face] = read_off([foldername,exemplars(kk).name]);
    
    foldersavebb=[foldersaveb(1:end-1),num2str(kk),'\'];
    
    mkdir(foldersavebb)
    for jj=1:length(listings)
        filename=[foldersavebb,num2str(kk),'-meshexem',num2str(jj),'.off'];
        write_off(filename, vertex, face);
        if mod(kk,100)==0
            disp(kk);
        end
    end
end

end