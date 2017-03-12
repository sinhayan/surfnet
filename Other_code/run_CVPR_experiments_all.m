function run_CVPR_experiments_all(Ns, synset, coord, loadgim, sz_out,prune,same,MTs,varargin)
% Usage example:  
%  run_cifar_experiments([20 32 44 56 110 164 1001], 'resnet', 'gpus', [1]);
% Options: 
%   'expDir'['exp'], 'gpus'[[]], 'border'[[4 4 4 4]], 
%   and more defined in cnn_cifar.m

setup;

opts.expDir = fullfile('data', 'CVPR_rigid');
opts.gpus = 1;
opts.preActivation = false;
opts = vl_argparse(opts, varargin); 

n_exp = numel(Ns); 
if ischar(MTs) || numel(MTs)==1, 
  if opts.preActivation, MTs='resnet-Pre'; end 
  if ischar(MTs), MTs = {MTs}; end; 
  MTs = repmat(MTs, [1, n_exp]); 
else
  assert(numel(MTs)==n_exp);
end

expRoot = opts.expDir; 

for i=1:n_exp, 
  opts.checkpointFn = @() plot_results(expRoot, ['CVPR',num2str(synset),'_',num2str(coord)],[],[], 'plots', {MTs{i}});
  if ~same
  opts.expDir = fullfile(expRoot, ...
    sprintf('CVPR%d_%d-%s-%d_az%d_el%d', synset,coord,MTs{i}, Ns(i),prune.azcase,prune.elcase)); 
  else
  opts.expDir = fullfile(expRoot, ...
    sprintf('RCVPR%d_%d-%s-%d_az%d_el%d_same%d', synset,coord,MTs{i}, Ns(i),prune.azcase,prune.elcase,same));         
  end
  [net,info] = res_CVPR_all(Ns(i),synset,coord,loadgim,sz_out,prune,same, 'modelType', MTs{i}, opts); 
end
