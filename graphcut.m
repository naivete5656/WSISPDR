addpath('./graphcut')
debug = false;

min_cell_size = 50;
min_hole_size = 10;
max_hole_size = Inf;
hole_min_perct_intensity = 0;
hole_max_perct_intensity = 100;
bp_thresh = 0.01;
bp_thresh2 = 0.001;
fill_holes_bool_oper = 'and';
manual_finetune = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
inpbasefolder = './output/guided/';
outbasefolder = './output/output/sef';
outfolder = [outbasefolder '/results/'];
outfolder2 = [outbasefolder '/labelresults/'];

infolders = dir(inpbasefolder);
mkdir(outfolder);
mkdir(outfolder2);
% mkdir(outfolder3);

for fileIndex=3:length(infolders)
    baseID = infolders(fileIndex).name;
    infolder = [inpbasefolder baseID '/'];
    infile = fullfile(infolder,'original.png');
    fcnfile = fullfile(infolder,'detection.png');
    posfile = fullfile(infolder,'peaks.txt');
    bpfolder = [infolder 'each_peak/'];
    bpfiles = dir([bpfolder '*.mat']);

    orgim = imread(infile);
    F = imread(fcnfile);
    orgim = double(orgim)/255;
    F = double(F)/255;

    fpos = readtable(posfile);
    fpos = fpos.Variables;
    fpos = fpos(:,[3 2]); % [y x]
    fpos = fpos(2:end,:);
    fpos(:,1) = fpos(:,1) + 1;
    fpos(:,2) = fpos(:,2) + 1;


    [Ny Nx] = size(orgim);
    Nz = length(bpfiles) - 1;

    if Nz <= 0;
        out = [outfolder baseID 'seg.tif'];
        imwrite(zeros(size(orgim)),out);
        out = [outfolder2 baseID 'segbp.tif'];
        imwrite(zeros(size(orgim)),out);
        out = [outfolder3 baseID 'label.tif'];
        imwrite(zeros(size(orgim)),out);

        continue;
    end

    BP = zeros(Ny,Nx,Nz);
    for fidx=2:Nz+1;
        bpfile = fullfile(bpfolder,bpfiles(fidx).name);
        bp = load(bpfile);
        bp = bp.image;
        BP(:,:,fidx-1) = double(bp)/255;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% preprocess
    % detect seed
    th_FCN = 0.91;
    maskF = zeros(size(F));
    maskF(F>th_FCN) = 1;
    
    BP(BP < bp_thresh) = 0;
    [bpm maxidx] = max(BP,[],3);
    maxidx(bpm==0) = 0;
    BPM = zeros(Ny,Nx,Nz);
    BPSM = zeros(Ny,Nx,Nz);
    tmp = zeros(Ny,Nx);
    
    for jj=1:Nz;
        tmp = tmp + exp(BP(:,:,jj));
    end
    
    mask = im2bw(max(BP,[],3),bp_thresh);
    for fidx=1:Nz;
        pos = fpos(fidx,:);% [y x]
        % max
        idx = find(maxidx~=fidx);
        bpm = BP(:,:,fidx);
        bpm(idx) = 0;
        BPM(:,:,fidx) = bpm;
        % softmax
        bpsm = exp(BP(:,:,fidx))./tmp.*im2bw(max(BP,[],3),bp_thresh);
        bpsm(find(BP(:,:,fidx)==0)) = 0;
        BPSM(:,:,fidx) = bpsm;

        a = zeros(Ny,Nx,3);  b = zeros(Ny,Nx,3);
        a(:,:,1) = orgim + bpm.^0.5;
        a(:,:,2) = orgim - bpm.^0.5;
        a(:,:,3) = orgim - bpm.^0.5;
        b(:,:,1) = orgim + bpsm;
        b(:,:,2) = orgim - bpsm;
        b(:,:,3) = orgim - bpsm;
    end

    RGB = label2rgb(maxidx,'jet','black','shuffle');


    %%%%%%%%%%%%%%%%%%%%%%
    %% GraphCut
    c0=2;
    A_th = 3;
    AL = zeros(size(orgim),'uint8');
    for fidx=1:Nz;
        pos = fpos(fidx,:);
        bpmf = BPM(:,:,fidx);
        BWf = im2bw(bpmf,bp_thresh2);
        BWf = imfill(BWf,'holes');
        C = bwconncomp(BWf);
        L = labelmatrix(C);
        Area = regionprops(L,'Area');
        Area = [Area(:).Area];
        inds = find(Area >= A_th);
        BWf = ismember(L,inds);
        if Nz == 1;
            BWb = zeros(size(BWf));
        else
            BWb = max(BPM(:,:,setdiff([1:Nz],fidx)),[],3);
        end
        BWb = im2bw(BWb,bp_thresh);
        BWb = imfill(BWb,'holes');
        if debug;
            figure(6);imshow(BWf);
        end  
        
        Dc1 = 0.4999*ones(size(orgim));
        Dc2 = 1 - Dc1;

        finds = find(BWf>0);
        Dc1(finds) = 1000000;
        Dc2(finds) = 0;
        binds = find(BWb>0);
        Dc2(binds) = 1000000;
        Dc1(binds) = 0;
        Dc = zeros(size(Dc1));
        Dc(:,:,1) = Dc1;
        Dc(:,:,2) = Dc2;

        Sc = [0 1;1 0];
        
        clear gch
        gch = GraphCut('open', Dc, 20*Sc, exp(-orgim*10), exp(-orgim*10));
        [gch L] = GraphCut('expand',gch);
        gch = GraphCut('close', gch);
        
        Ltmp = bwconncomp(L);

        ddd = [];
        for jj=1:Ltmp.NumObjects;
            ll = Ltmp.PixelIdxList{jj};
            [lly llx] = ind2sub(size(orgim),ll);
            lldd = sqrt(sum(([lly llx] - repmat(pos,length(lly),1))'.^2));
            ddd(jj) = min(lldd);
        end
        [mm id] = min(ddd);
        L = labelmatrix(Ltmp);
        L = ismember(L,id);
        L = imdilate(L,strel('disk',2));
        L = imfill(L,'holes');
        L = imerode(L,strel('disk',1));
        InterInds = find(L==1 & AL>0);
        interIDs = setdiff(unique(AL(InterInds)),0);
        nonInterInds = find(L==1 & AL==0);
        
        % separate the intersections
        if length(interIDs)>0;
            tmpInter = zeros(size(orgim));
            tmpInter(InterInds) = 1;
            [yi xi] = find(tmpInter>0);
            Rinter = [yi xi];
            winds = []; nwinds = [];
            for jj=1:length(interIDs)
                iid = interIDs(jj);
                ipos = fpos(iid,:);
                ddd1 = sqrt(sum((Rinter - repmat(ipos,size(Rinter,1),1))'.^2));
                ddd2 = sqrt(sum((Rinter - repmat(pos,size(Rinter,1),1))'.^2));
                dinds = find(ddd1>ddd2);
                ndinds = find(ddd2>=ddd1);
                R2inds = find(AL==iid);
                R2inds = union(setdiff(R2inds,InterInds),InterInds(ndinds));
                tmpR2 = zeros(size(orgim));
                tmpR2(R2inds) = 1;
                LR2 = bwlabel(tmpR2);
                R2id = LR2(ipos(1),ipos(2));
                LR2 = ismember(LR2,setdiff([1:max(LR2(:))],R2id));
                R1inds = find(LR2>0);
                nwinds = union(nwinds,setdiff(InterInds(ndinds),R1inds));
            end
            winds = setdiff(InterInds,nwinds);
            AL(winds) = fidx;
            AL(nonInterInds) = fidx;
        else
            AL(find(L==1)) = fidx;
        end
        if debug;
            figure(7);imshow(Dc1);
            figure(12);imagesc(AL);
        end
    end
    gcbw = boundarymask(AL);
    rgb = imoverlay(orgim,gcbw,[1 0 0]);
    rgb2 = imoverlay(RGB,gcbw,[1 0 0]);

    out = [outfolder baseID 'seg.tif'];
    imwrite(rgb,out);
    out = [outfolder baseID 'segbp.tif'];
    imwrite(rgb2,out);
    out = [outfolder2 baseID 'label.tif'];
    imwrite(AL,out);
end
1;



