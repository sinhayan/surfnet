
%1)read off files
%2)voxelize
%3)writeofffiles
%2)shape_clust
%3)reciprocal_mat
%4) run blended intrinsic maps
%5) plot_BLM_identify

addpath(genpath('Other_code'));
%%
%synsetname=02858304;sizen=96;
synsetname=02958343;size_vox=96; 

folder_shapenetsynset='H:\ShapeNetCore.v1\ShapeNetCore.v1';

folder_data=[folder_shapenetsynset,'\0',num2str(synsetname),'\0',num2str(synsetname),'\'];


%% voxelize
%read files from folder and voxelize, write them in a new folder



currentFolder = pwd;
folder_save=[currentFolder,'\Off_files_0',num2str(synsetname),'\'];
mkdir(folder_save);

voxelize(folder_data,size_vox,folder_save) 


%% create exemplars before running blended intrinsic maos
num_exem=4;
mesh_map=[currentFolder,'\Compiled_Binaries\'];
foldersave=[mesh_map,num2str(synsetname),'Off_blm\'];
foldersaveb=[mesh_map,num2str(synsetname),'Off_base\'];


shape_clust_CVPR(folder_save,num_exem,foldersave,foldersaveb);


listings=dir(folder_save);
listings=listings(3:end);
total_shapes=length(listings);

%% Blended intrinsic maps for exem to shape and exem to exem
%create commands for blended intrinsic maps

x = input('These commands need to be seperately run. Please download the Blended Intrinsic Maps. A copy of the code has been provided, but only runs on linux. Enter the folder and execute the code below. Acknolwedge their paper if you use this code. Press enter to continue');


folderself=[mesh_map,num2str(synsetname),'self\'];
mkdir(folderself);
command1=['for ((k=1;k<=',num2str(num_exem),';k++)); do for ((NUM=1;NUM<=',num2str(total_shapes),';NUM++));do sem -j+0 ./BlendedIntrinsicMapsMod ./',num2str(synsetname),'Off_base$k/$k-meshexem$NUM.off ./',num2str(synsetname),'Off_blm/$NUM-mesh.off ./',num2str(synsetname),'basemesh_rec; done ; done'];
command2=['for ((k=1;k<=',num2str(num_exem),';k++)); do for ((NUM=1;NUM<=',num2str(num_exem),';NUM++)); do sem -j+0 /mnt/d/Surfnet_code/Compiled_Binaries/BlendedIntrinsicMapsMod ./',num2str(synsetname),'Off_base/$k-meshexem.off ./',num2str(synsetname),'Off_base/$NUM-meshexem.off ./',num2str(synsetname),'self/basemesh_rec$k$NUM; done ; done'];

% only works for linux
% system(command1);
% system(command2);

%/mnt/d/Surfnet_code/Compiled_Binaries

%% Create goodness data


mesh_self_fold=[folderself,'\basemesh_rec'];

%mesh_map=['D:\Shape_correspondence_data\'];
parpool(num_exem);
parfor meshnum=1:num_exem
create_goodness_data_CVPR(meshnum,total_shapes,synsetname,num_exem,mesh_self_fold,foldersave,mesh_map,foldersaveb);
end

% mesh_fold=foldersave;
%  mesh_exem_fold=foldersaveb;
%  synset=synsetname;

%% Select mesh number based on goodness

select_each=zeros(total_shapes,num_exem);
for meshnum=1:num_exem
select_each(:,meshnum)=select_mesh_number_CVPR(meshnum,total_shapes,num_exem,synsetname);
end

count_exem=sum(select_each>0);
[~,meshbest]=max(count_exem);

%% create the imdb

imagefolder=[num2str(synsetname),'\'];
sz_images=128;
num_samples=78;
size_gim=128;
[fileimagesave,filegimsave]=create_gim_imdb_CVPR(meshbest,folder_save,imagefolder,sz_images,num_samples,synsetname,size_vox,size_gim,foldersaveb,mesh_self_fold,foldersave,mesh_map);
 

%% learning 

x = input('Please download matconvnet for running the learning codes');
%Learning code coming soon


