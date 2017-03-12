function [fileimagesave,filegimsave]=create_gim_imdb_CVPR(meshnum,filename,imagefolder,sz_images,num_samples,synset,sz_voxel,sizen,mesh_exem_fold,mesh_self_fold,mesh_fold,mesh_map)
%rotate according to image
addpath(genpath('learning_geometry_images-master'));
addpath(genpath('files'));


load([num2str(synset),'Base_save',num2str(meshnum)]);


% filename='H:\ShapeNetCore.v1\ShapeNetCore.v1\02691156\Off_files\';
% imagefolder='H:\2Dto3D\02691156\';

flipidx=1;
listfiles=dir(filename);
listfiles=listfiles(3:end);

idx_files=find(select_meshnumber);


listconsider=listfiles(idx_files);

% sz_images=128;
% num_samples=78;
%read images
im_check=zeros(length(listconsider),1);
im_all=cell(length(listconsider),1);

for kk=1:length(listconsider)
    foldername=listconsider(kk).name;
    foldername=foldername(1:end-4);
    foldername_search=[ imagefolder,foldername];
    
    listimages=dir([foldername_search '/*.png']);
    
    if length(listimages)<num_samples
        disp(kk)
        disp('not all images are there');
        im_check(kk)=1;
    end
    im_mat=zeros(sz_images,sz_images,3,num_samples,'uint8');
    
    
    for pp=1:length(listimages)
        imagename=[foldername_search,'/',listimages(pp).name];
        im=imread(imagename);
        sz_img=size(im);
        ijk=find(im);
        [i,j,~]=ind2sub(sz_img,ijk);
        
        if length(sz_img)==3 && sz_img(3)==3
            min_x=min(i);
            max_x=max(i);
            min_y=min(j);
            max_y=max(j);
            diff_x=max_x-min_x+1;
            diff_y=max_y-min_y+1;
            if (diff_x==sz_images || diff_y==sz_images) && (max(diff_x,diff_y)<sz_images)
                im_mat(:,:,:,pp)=im;
            else
                %                 if max(diff_x,diff_y)<min(sz_img(1),sz_img(2))
                if diff_x>diff_y
                    diffres=diff_x-diff_y;
                    evenodd=mod(diffres,2);
                    im_temp=im(min_x:max_x,min_y:max_y,:);
                    if evenodd
                        
                        pad1=zeros(diff_x,floor(diffres/2),3);
                        pad2=zeros(diff_x,floor(diffres/2)+1,3);
                        im_temp=cat(2,pad1,im_temp,pad2);
                        im_temp=imresize(im_temp,[sz_images,sz_images]);
                        im_mat(:,:,:,pp)=im_temp;
                        
                    else
                        pad1=zeros(diff_x,floor(diffres/2),3);
                        pad2=zeros(diff_x,floor(diffres/2),3);
                        im_temp=cat(2,pad1,im_temp,pad2);
                        im_temp=imresize(im_temp,[sz_images,sz_images]);
                        im_mat(:,:,:,pp)=im_temp;
                        
                    end
                    
                    
                elseif diff_x<diff_y
                    diffres=diff_y-diff_x;
                    evenodd=mod(diffres,2);
                    im_temp=im(min_x:max_x,min_y:max_y,:);
                    if evenodd
                        
                        pad1=zeros(floor(diffres/2),diff_y,3);
                        pad2=zeros(floor(diffres/2)+1,diff_y,3);
                        im_temp=cat(1,pad1,im_temp,pad2);
                        im_temp=imresize(im_temp,[sz_images,sz_images]);
                        im_mat(:,:,:,pp)=im_temp;
                        
                    else
                        pad1=zeros(floor(diffres/2),diff_y,3);
                        pad2=zeros(floor(diffres/2),diff_y,3);
                        im_temp=cat(1,pad1,im_temp,pad2);
                        im_temp=imresize(im_temp,[sz_images,sz_images]);
                        im_mat(:,:,:,pp)=im_temp;
                        
                    end
                else
                    im_temp=im(min_x:max_x,min_y:max_y,:);
                    im_temp=imresize(im_temp,[sz_images,sz_images]);
                    im_mat(:,:,:,pp)=im_temp;
                    
                end
                
                
            end
            
        else
            disp(kk)
            disp(pp)
            disp('not a RGB image');
            im_check(kk)=1;
        end
        
        
        
    end
    
    im_all{kk}=uint8(im_mat);
    if mod(kk,50)==0
        disp(kk);
    end
    
end

num_samplesi=num_samples;
if flipidx
    num_samplesi=2*num_samples;
end

im_imdb=zeros(sz_images,sz_images,3,num_samplesi*length(listconsider),'uint8');

count=1;
for kk=1:length(listconsider)
    
    im=uint8(im_all{kk});
    if flipidx
        im=cat(4,im,fliplr(im));
    end
    im_imdb(:,:,:,count:count+num_samplesi-1)=im;
    count=count+num_samplesi';
    if mod(kk,100)==0
        disp(kk);
    end
end

fileimagesave=[num2str(synset),'ImagesIMDBHQ_',num2str(meshnum)];
save(fileimagesave,'im_imdb','-v7.3');
clear im_all
%%

% mesh_exem_fold='C:\Users\sinha\Downloads\Compiled_Binaries\Off_base\';
% mesh_self_fold='C:\Users\sinha\Downloads\Compiled_Binaries\self\basemesh_rec';
% mesh_fold='C:\Users\sinha\Downloads\Compiled_Binaries\Off_blm\';
% mesh_map='D:\Shape_correspondence_data\';


maxvert=5000;
hier=0;
iter=300;
% sizen=128;


[v1,f1]=read_off([mesh_exem_fold,num2str(meshnum),'-meshexem.off']);
v1=v1';
f1=f1';
facen=f1;
sph_verts=DeepLearning_param(v1,f1,maxvert,hier,iter);



%create gometry images

%create transformation matrix

num_az=13;
deg=linspace(0,180,num_az);
el=[-15,-30,0,15,30,45];
num_el = length(el);
count=1;
M_cell=cell(num_samples,1);
for aza=1:num_az
    for ela=1:num_el
        
        Mx=AxelRot(el(ela),[1,0,0]);
        Mz=AxelRot(deg(aza),[0,0,1]);
        %        M=Mx*Mz;
        M=Mz*Mx;
        
        M_cell{count}=M;
        count=count+1;
    end
end

% account for noise in gim
select_meshnumber_idx=select_meshnumber(idx_files);
gim_all=cell(length(listconsider),1);


residual_vert=cell(num_samples,1);
v_center=[sz_voxel/2,sz_voxel/2,sz_voxel/2];
ptCloud = pointCloud(v1);
ptCloudcenter = pointCloud(v_center);
for zz =1:num_samples
        ptCloudOut = pctransform(ptCloud,affine3d(M_cell{zz}));
        verts_rot =ptCloudOut.Location;
       
        ptCloudOut = pctransform(ptCloudcenter,affine3d(M_cell{zz}));
        verts_rotcen =ptCloudOut.Location;  
        res_vert=verts_rot-verts_rotcen;
        
        residual_vert{zz}= perform_sgim_sampling_fast(res_vert', sph_verts', facen', sizen);
end

fileressave=[num2str(synset),'GeomResidualIMDBN_',num2str(meshnum)];
save(fileressave,'residual_vert','-v7.3');

synsetname=synset;
smooth=1;
%parpool(8);
parfor kkk=1:length(listconsider)
    pp=idx_files(kkk);
    [v,f]=read_off([mesh_fold,num2str(pp),'-mesh.off']);
    
    v=v';
    
    if smooth
    v = perform_mesh_smoothing(f,v,v);
    end
    f=f';
    
    v_center=[sz_voxel/2,sz_voxel/2,sz_voxel/2];
    v_cell=cell(num_samples,1);
    
    
    %   norm_cell=zeros(num_samples,1);
    ptCloud = pointCloud(v);
    ptCloudcenter = pointCloud(v_center);
    
    for zz =1:num_samples
        ptCloudOut = pctransform(ptCloud,affine3d(M_cell{zz}));
        verts_rot =ptCloudOut.Location;
       
        ptCloudOut = pctransform(ptCloudcenter,affine3d(M_cell{zz}));
        verts_rotcen =ptCloudOut.Location;  
        v_cell{zz}=verts_rot-verts_rotcen;
    end
    
    
    
    
    gim_mat=zeros(sizen,sizen,3,num_samples);
    if select_meshnumber_idx(kkk)==meshnum
        filename=[mesh_map,num2str(synsetname),'basemesh_rec\',num2str(meshnum),'-meshexem',num2str(pp),'\Blended_DenseMapPreceise_',num2str(meshnum),'-meshexem',num2str(pp),'_to_',num2str(pp),'-mesh.dense.preceise.map'];

        xmat=load(filename);
        face_num=f(xmat(:,1)+1,:);
        
        for zz =1:num_samples
            vert=xmat(:,2).*v_cell{zz}(face_num(:,1),:)+xmat(:,3).*v_cell{zz}(face_num(:,2),:)+xmat(:,4).*v_cell{zz}(face_num(:,3),:);
            gim_mat(:,:,:,zz) = perform_sgim_sampling_fast(vert', sph_verts', facen', sizen);
        end
        gim_all{kkk}=single(gim_mat);
    else
        meshnum2=select_meshnumber_idx(kkk);
        filename=[mesh_map,num2str(synsetname),'basemesh_rec\',num2str(meshnum2),'-meshexem',num2str(pp),'\Blended_DenseMapPreceise_',num2str(meshnum2),'-meshexem',num2str(pp),'_to_',num2str(pp),'-mesh.dense.preceise.map'];
       
        xmat=load(filename);
        face_num=f(xmat(:,1)+1,:);
        
        
        filename=[mesh_self_fold,num2str(meshnum),num2str(meshnum2),'\',num2str(meshnum),'-meshexem\Blended_DenseMapPreceise_',num2str(meshnum),'-meshexem_to_',num2str(meshnum2),'-meshexem.dense.preceise.map'];
        
        xmat2=load(filename);
        
        [v2,f2]=read_off([mesh_exem_fold,num2str(meshnum2),'-meshexem.off']);
        
        f2=f2';
        face_num2=f2(xmat2(:,1)+1,:);
        
        for zz =1:num_samples
            vertt=xmat(:,2).*v_cell{zz}(face_num(:,1),:)+xmat(:,3).*v_cell{zz}(face_num(:,2),:)+xmat(:,4).*v_cell{zz}(face_num(:,3),:);     
            vertn=xmat2(:,2).*vertt(face_num2(:,1),:)+xmat2(:,3).*vertt(face_num2(:,2),:)+xmat2(:,4).*vertt(face_num2(:,3),:);
            gim_mat(:,:,:,zz) = perform_sgim_sampling_fast(vertn', sph_verts', facen', sizen);
        end
        gim_all{kkk}=single(gim_mat);
        
    end
    
    if mod(kkk,10)==0
        disp(pp);
    end
    
    
end


gim_imdb=zeros(sizen,sizen,3,num_samplesi*length(listconsider),'single');
count=1;
for kk=1:length(listconsider)    
   
    gim=single(gim_all{kk});
     if flipidx
        gimt=gim;
        gimt(:,:,1,:)=-gimt(:,:,1,:);
        gim=cat(4,gim,gimt);
     end
    
    gim_imdb(:,:,:,count:count+num_samplesi-1)=gim;
    count=count+num_samplesi';
    if mod(kk,100)==0
        disp(kk);
    end 
end


filegimsave=[num2str(synset),'GeomImagesIMDBN_',num2str(meshnum)];
save(filegimsave,'gim_imdb','-v7.3');
clear gim_all gim_imdb









end