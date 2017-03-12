function [net, info] = res_CVPR_all(m,synset,coord,loadgim, sz_out,prune, same,varargin)
% res_cifar(20, 'modelType', 'resnet', 'reLUafterSum', false,...
% 'expDir', 'data/exp/cifar-resNOrelu-20', 'gpus', [2])
% setup;
opts.modelType = 'resnet' ;
opts.preActivation = false;
opts.reLUafterSum = false;
opts.shortcutBN = false;
[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.preActivation
    opts.expDir = fullfile('exp', ...
        sprintf('CVPR%d_%d-%s-%d_az%d_el%d', synset,coord,opts.modelType,m,prune.azcase,prune.elcase)) ;
else
    opts.expDir = fullfile('exp', ...
        sprintf('CVPR%d_%d-resnet-Pre-%d_az%d_el%d_same%d',synset,coord,m,prune.azcase,prune.elcase,same)) ;
end


opts.dataDir = fullfile('data','CVPR') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');

opts.gpus = [];
opts.checkpointFn = [];
opts = vl_argparse(opts, varargin) ;
savePath = fullfile(opts.expDir);
mkdir(savePath);
save(savePath,'prune');


% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

if opts.preActivation
    net = res_CVPR_preactivation_init(m);
else
    net = res_CVPR_initall(m,coord,sz_out, 'networkType', opts.modelType, ...
        'reLUafterSum', opts.reLUafterSum) ;
end

foldsave='D:\Shape_correspondence_data\';
%foldsave='C:\Users\sinha\Downloads\map_synchronization-master\map_synchronization-master\consistent_shape_maps\';


suffix=1;
if synset==2858304
    suffix=4;
end

imdb_matfile_in=([foldsave,num2str(synset),'ImagesIMDBHQ_',num2str(suffix)]);
imdb_matfile_in=load(imdb_matfile_in);
[szi_1,szi_2,szi_3,sz_data]=size(imdb_matfile_in.im_imdb);
imdb_matfile_in=imdb_matfile_in.im_imdb;

imdb_matfile_out=([foldsave,num2str(synset),'GeomImagesIMDBHQ_',num2str(suffix),'_',num2str(sz_out) ]);
if ~loadgim
    imdb_matfile_out=matfile(imdb_matfile_out);
    [szo_1,szo_2,~,~]=size(imdb_matfile_out,'gim_imdb');
    szo_3=length(num2str(coord));
else
    imdb_matfile_out=load(imdb_matfile_out);
    imdb_matfile_out=imdb_matfile_out.gim_imdb;
    [szo_1,szo_2,~,~]=size(imdb_matfile_out);
    szo_3=length(num2str(coord));
    if szo_3==1
        imdb_matfile_out=imdb_matfile_out(:,:,coord,:);
    end
end

if szo_3==3
    coord=[1,2,3];
end


numsamples=156;

test_num=numsamples*round(0.2*sz_data/numsamples);

set=ones(1,sz_data);
set(end:-1:end-test_num+1)=3;

if prune.prune
    if ~loadgim
        num_items=sz_data/numsamples;
        prunemat=prune.prunevec*ones(1,num_items);
        prunemat=prunemat(:);
        prunemat=(prunemat'>0);
        set=set.*prunemat;
    else
        num_items=sz_data/numsamples;
        prunemat=prune.prunevec*ones(1,num_items);
        prunemat=prunemat(:);
        prunemat=(prunemat'>0);
        imdb_matfile_in=imdb_matfile_in(:,:,:,prunemat);
        imdb_matfile_out=imdb_matfile_out(:,:,:,prunemat);
        set=set(prunemat);
    end
end

imdb.images.set=set;

imdb.images.data=imdb_matfile_in;
clear imdb_matfile_in
imdb.images.labels=imdb_matfile_out;
clear imdb_matfile_out
imdb.images.input_size=[szi_1,szi_2,szi_3];
imdb.images.output_size=[szo_1,szo_2,szo_3];
imdb.images.numsamples=numsamples;
imdb.images.numsampload=sum(prune.prunevec);
imdb.meta.loadgim=loadgim;
imdb.meta.coord=coord;
imdb.meta.same=same;


if prune.elcase==1
    sameidx=3;
elseif prune.elcase==2
    sameidx=2;
elseif prune.elcase==3
    sameidx=2;
elseif prune.elcase==4
    sameidx=1;
elseif prune.elcase==5
    sameidx=1;
end
imdb.meta.sameidx=sameidx;

% load('gim_boat');
% imdb.resvert=gim_boat(:,:,coord);
% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train_dag_check;

rng('default');
[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    'gpus', opts.gpus, ...
    'val', find(imdb.images.set == 3), ...
    'derOutputs', {'loss', 1}, ...
    'checkpointFn', opts.checkpointFn) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
imdb_matfile_in=imdb.images.data;
images=imdb_matfile_in(:,:,:,batch);

imdb_matfile_out=imdb.images.labels;
coord=imdb.meta.coord;
loadgim=imdb.meta.loadgim;
numsamples=imdb.images.numsamples;
sameidx=imdb.meta.sameidx;
same=imdb.meta.same;
if ~loadgim
    labels=zeros([imdb.images.output_size,length(batch)],'single');
    if ~same
        
        for pp=1:length(batch)
            labels(:,:,:,pp)=imdb_matfile_out.gim_imdb(:,:,coord,batch(pp));
        end
    else
        for pp=1:length(batch)
            if mod(batch(pp),numsamples)~=0
                choice=floor(batch(pp)/numsamples)*numsamples+3;
            else
                choice=((batch(pp)/numsamples)-1)*numsamples+3;
            end
            labels(:,:,:,pp)=imdb_matfile_out.gim_imdb(:,:,coord,choice);
        end
    end
else
    if ~same
        labels=imdb_matfile_out(:,:,:,batch);
    else
        choice=zeros(1,length(batch));
        numsamples=imdb.images.numsampload;
        for pp=1:length(batch)
            if mod(batch(pp),numsamples)~=0
                choice(pp)=floor(batch(pp)/numsamples)*numsamples+sameidx;
            else
                choice(pp)=((batch(pp)/numsamples)-1)*numsamples+sameidx;
            end
        end
        labels=imdb_matfile_out(:,:,:,choice);
    end
end

% resvert=imdb.resvert;
% labels=labels-resvert;
% images=single(images);
% labels=single(labels)*100;

 images=single(images);
 labels=single(labels)*10;


if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'image', images, 'label', labels} ;

