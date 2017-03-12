function select_meshnumber=select_mesh_number_CVPR(meshnum,num_shapes,base_num,synset)


filenamesave=[num2str(synset),'Basemesh',num2str(meshnum)];
load(filenamesave);
load(filenamesavegim);
load(filenamesavegimother);
Dother_all(1:num_shapes,meshnum)=D_all(:);
gimother_all(1:num_shapes,meshnum)=gim_all(:);

clear D_all
clear gim_all





%These values need to be tailored
thresh_xsaf=0.3;
thresh_ysaf=0.2;
thresh_zsaf=0.45;

thresh_distfac=[0.15 0.15 0.15];
thresh_dist=1.25;


select_meshnumber=zeros(num_shapes,1);
for kk=1:num_shapes
    gimcheck=1;
    for pp=1:base_num
        if isempty(gimother_all{kk,pp})
            gimcheck=0;
            break;
        end
    end
    
    if gimcheck
        %first identify is parametrization is consistent
        x_max=max(max(gim_org(:,:,1)));
        x_min=min(min(gim_org(:,:,1)));
        y_max=max(max(gim_org(:,:,2)));
        y_min=min(min(gim_org(:,:,2)));
        z_max=max(max(gim_org(:,:,3)));
        z_min=min(min(gim_org(:,:,3)));
        
        
        safety_factor=zeros(base_num,3);
        dist_factor=zeros(base_num,3);
        dist_mesh=zeros(1,base_num);
        for pp=1:base_num
            gim1=gimother_all{kk,pp};
            
            x_max2=max(max(gim1(:,:,1)));
            x_min2=min(min(gim1(:,:,1)));
            y_max2=max(max(gim1(:,:,2)));
            y_min2=min(min(gim1(:,:,2)));
            z_max2=max(max(gim1(:,:,3)));
            z_min2=min(min(gim1(:,:,3)));
            
            
            
            
            gim_dist=sqrt(((gim1-gim_org).^2));
            gim_distx=gim_dist(:,:,1);
            gim_disty=gim_dist(:,:,2);
            gim_distz=gim_dist(:,:,3);
            
            x_maxgim=max(gim_distx(:));
            y_maxgim=max(gim_disty(:));
            z_maxgim=max(gim_distz(:));
            
            
            
            
            
            
            
            x_corrdistg=max(abs(x_max-x_min2),abs(x_min-x_max2));
            y_corrdistg=max(abs(y_max-y_min2),abs(y_min-y_max2));
            z_corrdistg=max(abs(z_max-z_min2),abs(z_min-z_max2));
            
            x_safety_factorg=x_maxgim/x_corrdistg;
            y_safety_factorg=y_maxgim/y_corrdistg;
            z_safety_factorg=z_maxgim/z_corrdistg;
            
            
            safety_factor(pp,:)=[x_safety_factorg,y_safety_factorg,z_safety_factorg];
            
            
            dist_factor(pp,1)=mean(gim_distx(:))/x_corrdistg;
            dist_factor(pp,2)=mean(gim_disty(:))/y_corrdistg;
            dist_factor(pp,3)=mean(gim_distz(:))/z_corrdistg;
            
            dist_mesh(pp)=mean(Dother_all{kk,pp});
            
            
        end
        
        idxx=safety_factor(:,1)<thresh_xsaf;
        idxy=safety_factor(:,2)<thresh_ysaf;
        idxz=safety_factor(:,3)<thresh_zsaf;
        
        idxs=(idxx.*idxy.*idxz)>0;
        num_mesh_select=1:base_num;
        num_mesh_select=num_mesh_select(idxs);
        if ~isempty(num_mesh_select)
            
            safety_factor_select=mean(safety_factor(idxs,:),2);
            
            if length(safety_factor_select)==1
                dist_curr=dist_factor(idxs,:)<thresh_distfac;
                if sum(dist_curr)==3
                    iddist=dist_mesh(idxs);
                    if iddist<thresh_dist
                        select_meshnumber(kk)=num_mesh_select;
                    end
                end
            elseif length(safety_factor_select)>1
                
                
                dist_curr=dist_factor(idxs,:)<thresh_distfac;
                dist_curr=sum(dist_curr,2);
                num_mesh_select(dist_curr<3)=[];
                iddist=dist_mesh(num_mesh_select);
                num_mesh_select(iddist>thresh_dist)=[];
                if length(num_mesh_select)==1
                    
                    select_meshnumber(kk)=num_mesh_select;
                elseif length(num_mesh_select)>1
                    safety_factor_select=mean(safety_factor(num_mesh_select,:),2);
                    [~,pref_curr]=sort(safety_factor_select);
                    pref_curr=num_mesh_select(pref_curr);
                    dist_factor_select=mean(dist_factor(num_mesh_select,:),2);
                    [~,pref_currd]=sort(dist_factor_select);
                    pref_currd=num_mesh_select(pref_currd);
                    
                    if isequal(pref_curr,pref_currd)
                        distcomp=dist_mesh(num_mesh_select);
                        [vald,sortidx]=sort(distcomp,'ascend');
                        if vald(1)*1.1<vald(2)
                            pref_order=num_mesh_select(sortidx);
                            select_meshnumber(kk)=pref_order(1);
                        else
                            pref_order=num_mesh_select(sortidx);
                            if isequal(pref_curr,pref_order)
                                select_meshnumber(kk)=pref_order(1);
                            else
                                select_meshnumber(kk)=pref_curr(1);
                            end
                            
                        end
                        
                    else
                        
                        distcomp=dist_mesh(num_mesh_select);
                        [~,sortidx]=sort(distcomp,'ascend');
                        pref_order=num_mesh_select(sortidx);
                        select_meshnumber(kk)=pref_order(1);
                        
                    end
                    
                    
                end
            end
            
            
        else
            disp(kk)
            disp('not selected');
        end
        
    end
       
end

% filesave=[num2str(synset),'Base_save',num2str(meshnum)];
% save(filesave,'select_meshnumber');

end