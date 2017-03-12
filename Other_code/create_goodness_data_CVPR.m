function create_goodness_data_CVPR(meshnum,total_shapes,synset,num_exem,mesh_self_fold,mesh_fold,mesh_map,mesh_exem_fold)


% mesh_fold=foldersave;
% mesh_exem_fold=foldersaveb;
% synset=synsetname;


%  mesh_exem_fold='C:\Users\sinha\Downloads\Compiled_Binaries\Off_base\';
% mesh_self_fold='C:\Users\sinha\Downloads\Compiled_Binaries\self\basemesh_rec';
% mesh_fold='C:\Users\sinha\Downloads\Compiled_Binaries\Off_blm\';
% mesh_map='D:\Shape_correspondence_data\';

num_shapes=total_shapes;
addpath(genpath('learning_geometry_images-master'));
addpath(genpath('files'));


sizen=64;
maxvert=5000;
hier=0;
iter=300;
vert_all=cell(num_shapes,1);
gim_all=cell(num_shapes,1);
gimother_all=cell(num_shapes,num_exem);
D_all=cell(num_shapes,1);
Dother_all=cell(num_shapes,num_exem);
filenamesavegim=[num2str(synset),'gim_all',num2str(meshnum)];
save(filenamesavegim,'gim_all','-v7.3');
filenamesavegimother=[num2str(synset),'gimother_all',num2str(meshnum)];
save(filenamesavegimother,'gimother_all','-v7.3');


synsetname=synset;

[v1,f1]=read_off([mesh_exem_fold,num2str(meshnum),'-meshexem.off']);
v1=v1';
f1=f1';
facen=f1;
sph_verts=DeepLearning_param(v1,f1,maxvert,hier,iter);


gim_org = perform_sgim_sampling(v1', sph_verts', f1', sizen);

gimother_all_matfile=matfile(filenamesavegimother,'Writable',true);
gim_all_matfile=matfile(filenamesavegim,'Writable',true);

for pp=1:num_shapes
    [v,f]=read_off([mesh_fold,num2str(pp),'-mesh.off']);
    v=v';
    f=f';
    
    vert_all{pp}=v;
    
    
    filename=[mesh_map,num2str(synsetname),'basemesh_rec\',num2str(meshnum),'-meshexem',num2str(pp),'\Blended_DenseMapPreceise_',num2str(meshnum),'-meshexem',num2str(pp),'_to_',num2str(pp),'-mesh.dense.preceise.map'];
    if exist(filename,'file')
        
        xmat=load(filename);
        face_num=f(xmat(:,1)+1,:);
        vert=xmat(:,2).*v(face_num(:,1),:)+xmat(:,3).*v(face_num(:,2),:)+xmat(:,4).*v(face_num(:,3),:);
        gim = perform_sgim_sampling_fast(vert', sph_verts', facen', sizen);
        gim_vert=reshape(gim,(size(gim,1)).^2,3);
        [~,D] = knnsearch(gim_vert,v);
        D_all{pp}=D;
        gim_all_matfile.gim_all(pp,1)={single(gim)};
        
    end
    
    
    
    meshvec=setdiff(1:num_exem,meshnum);
    
    for meshnum2=meshvec
        filename=[mesh_map,num2str(synsetname),'basemesh_rec\',num2str(meshnum2),'-meshexem',num2str(pp),'\Blended_DenseMapPreceise_',num2str(meshnum2),'-meshexem',num2str(pp),'_to_',num2str(pp),'-mesh.dense.preceise.map'];
        
        if exist(filename,'file')
            
            xmat=load(filename);
            face_num=f(xmat(:,1)+1,:);
            vertt=xmat(:,2).*v(face_num(:,1),:)+xmat(:,3).*v(face_num(:,2),:)+xmat(:,4).*v(face_num(:,3),:);
            
            
            filename=[mesh_self_fold,num2str(meshnum),num2str(meshnum2),'\',num2str(meshnum),'-meshexem\Blended_DenseMapPreceise_',num2str(meshnum),'-meshexem_to_',num2str(meshnum2),'-meshexem.dense.preceise.map'];
            if exist(filename,'file')
                xmat=load(filename);
                
                [v2,f2]=read_off([mesh_exem_fold,num2str(meshnum2),'-meshexem.off']);
                
                v2=v2';
                f2=f2';
                face_num=f2(xmat(:,1)+1,:);
                
                vertn=xmat(:,2).*vertt(face_num(:,1),:)+xmat(:,3).*vertt(face_num(:,2),:)+xmat(:,4).*vertt(face_num(:,3),:);
                gim1 = perform_sgim_sampling_fast(vertn', sph_verts', facen', sizen);
                gim1_vert=reshape(gim1,(size(gim1,1)).^2,3);
                [~,D1] = knnsearch(gim1_vert,v);
                
                Dother_all{pp,meshnum2}=D1;
                gimother_all_matfile.gimother_all(pp,meshnum2)={single(gim1)};
            end
        end
    end
    
    if mod(pp,10)==0
        disp(pp);
    end
    
end

filenamesave=[num2str(synset),'Basemesh',num2str(meshnum)];
%save(filenamesave,'vert_all','D_all','gim_all','Dother_all','gimother_all','gim_org','sph_verts','-v7.3');
save(filenamesave,'vert_all','D_all','Dother_all','gim_org','sph_verts','filenamesavegimother','filenamesavegim','-v7.3');

end