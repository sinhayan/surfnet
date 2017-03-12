function voxelize(folder_data,sizen,folder_save) 

currentFolder = pwd;
folder_dep=[currentFolder,'\learning_geometry_images-master\'];
addpath(genpath(folder_dep))


Files_model=dir(fullfile(folder_data)) ;
Files_model=Files_model(3:end);


% Parameters

perturb=1;
smooth=1;


% Loop over all files


%parpool(8);
parfor ii=1:length(Files_model)
    disp(ii);
    %Read files
    filename=[folder_data,Files_model(ii).name,'/model.obj'];
    [V,F] = readOBJ(filename);

    
    V=V(:,[1,3,2]);  %change coordinate axis

   
    % Function to make mesh accurate and genus zero
    [facen,vertn,considered]=voxelelize_genus_CVPR(V,F,sizen,perturb);
    

        
    if considered
        filesave=[folder_save,Files_model(ii).name,'.off'];
        
            if smooth
%        vertn  = laplacian_smooth(vertn,facen,'cotan',[],0.1);
       vertn = perform_mesh_smoothing(facen,vertn,vertn);
            end
    
        write_off(filesave, vertn, facen);
    end
    

        
end   