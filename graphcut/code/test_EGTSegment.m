addpath('C:\Users\bise-r\Desktop\code\toolbox\Kernel_GraphCuts\GCMex-master\')
addpath('C:\Users\bise-r\Desktop\code\toolbox\DRLSE_v0\')
debug = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% setting
% basefolder = '..\graphcut_data\';
basefolder = '..\guided_backpropagation\test\';
infolders = dir(basefolder);
min_cell_size = 100;
min_hole_size = 50;
max_hole_size = Inf;
hole_min_perct_intensity = 0;
hole_max_perct_intensity = 100;
fill_holes_bool_oper = 'and';
manual_finetune = -5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% segments foreground
for fidx=3:length(infolders);
    baseID = infolders(fidx).name;
    infolder = [basefolder baseID '\'];
%     infile = fullfile(infolder,'original_image.tif');
    infile = fullfile(infolder,'original.tif');
    orgim = imread(infile);
    S = EGT_Segmentation(orgim, min_cell_size, min_hole_size, max_hole_size, hole_min_perct_intensity, hole_max_perct_intensity, fill_holes_bool_oper, manual_finetune);
    
%     bpfolder = [basefolder 'each_peak_backward\'];
    bpfolder = [basefolder 'each_peak\'];
    bpfiles = dir([bpfolder '*.tif']);
    [Ny Nx] = size(orgim);
    Nz = length(bpfiles);
    BP = zeros(Ny,Nx,Nz);
    for fidx=1:length(bpfiles);
        bpfile = fullfile(bpfolder,bpfiles(fidx).name);
        bp = imread(bpfile);
        BP(:,:,fidx) = double(bp)/255;
    end
    
    figure(1);imshow(orgim);
%     figure(2);imshow(S);
    
    gcbw = boundarymask(S);
    rgb = imoverlay(orgim,gcbw,[1 0 0]);
    figure(3);imshow(rgb);
end


