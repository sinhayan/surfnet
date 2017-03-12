azcase=1;
elcase=5;
if elcase==1
el=[1,1,1,1,1,1]; %all
elseif elcase==2
 el=[1,0,1,1,1,0]; %some
elseif elcase==3
 el=[1,0,1,1,0,0]; %equal
elseif elcase==4
 el=[0,0,1,1,0,0]; %two
elseif elcase==5
 el=[0,0,1,1,1,1]; %two
else
    el=[];
end
el=el';

if azcase==1
az1=[1 1 1 1 1 1 1 1 1 1 1 1 1];%all
az2=[0 1 1 1 1 1 1 1 1 1 1 1 0];
elseif azcase==2
 az1=[1 0 1 0 1 0 1 0 1 0 1 0 1]; %30 degrees
 az2=[0 0 1 0 1 0 1 0 1 0 1 0 0];
elseif azcase==3
% az1=[1 0 0 1 0 0 1 0 0 1 0 0 1];%45 degrees
% az2=[0 0 0 1 0 0 1 0 1 1 0 0 0];
elseif azcase==4
az1=[1 0 0 0 1 0 0 0 1 0 0 0 1]; % %60 degrees
az2=[0 0 0 0 1 0 0 0 1 0 0 0 0];
else
    az1=[];
    az2=[];
end

az=[az1,az2];
prune.prune=1;
matelaz=ones(length(el),1)*az;
prunevec=bsxfun(@times,matelaz,el);
prune.prunevec=prunevec(:);
prune.azcase=azcase;
prune.elcase=elcase;

%%

loadgim=0;
m=32;
gpuid=2;
sz_out=128;

synset=2958343;
same=0;
%synset=2958343;
%synset=2691156;

for coord=1:3
run_CVPR_experiments_all(m,synset,coord,loadgim, sz_out,prune,same,'resnet','gpus',gpuid);
end
