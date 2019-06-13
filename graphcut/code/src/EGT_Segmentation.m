% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.



% S = EGT_Segmentation(I, min_cell_size, lower_hole_size_bound, upper_hole_size_bound, manual_finetune)
% 
% Inputs
%     -I: image to be segmented using automatic gradient based edge detection.
%     -min_cell_size: smallest object in segmented image that will be kept. Smaller objects will be removed.
%     -lower_hole_size_bound: holes smaller than this in the segmented image will be filled in.
%     -upper_hole_size_bound: holes larger than this in the segmented image will be filled in.
%     -manual_finetune: adjustment value to be added to the computed percentile threshold
% 
% Outputs
%     -S: segmented image (logical)

function S = EGT_Segmentation(I, min_cell_size, min_hole_size, max_hole_size, hole_min_perct_intensity, hole_max_perct_intensity, fill_holes_bool_oper, manual_finetune)
if ~exist('min_cell_size','var')
    min_cell_size = 1;
end
if ~exist('min_hole_size','var')
    min_hole_size = Inf;
end
if ~exist('max_hole_size','var')
    max_hole_size = Inf;
end
if ~exist('hole_min_perct_intensity','var')
    hole_min_perct_intensity = 0;
end
if ~exist('hole_max_perct_intensity','var')
    hole_max_perct_intensity = 100;
end
if ~exist('fill_holes_bool_oper', 'var')
  fill_holes_bool_oper = 'AND';
end
if ~exist('manual_finetune','var')
    manual_finetune = 0;
end

% this controls how far each increment of manual_finetune moves the percentile threshold
greedy_step = 1; 

% Compute image gradient and percentiles
S = imgradient(single(I));
S1 = nonzeros(S);

% Compute gradient image histogram
ratio = (max(S1)-min(S1))/1000;
% factor = ceil(numel(S1)/25000000); % set the factor so limit the number of pixel to 25 million
% [hist_data,~] = hist(S1(1:factor:end),min(S1):ratio:max(S1));
[hist_data,~] = hist(S1(:),min(S1):ratio:max(S1));


% get the mode and the corresponding frequency value
[~,hist_mode_loc] = sort(hist_data, 'descend');
hist_mode_loc = round(mean(hist_mode_loc(1:3))); % take the average of the first 3 peaks
% normalize the histogram counts between 0-1
temp_hist = hist_data/sum(hist_data)*100;
% compute lower bound
lower_bound = 3*hist_mode_loc;
if lower_bound > numel(temp_hist)
    warning('lower bound set to end of list.');
    lower_bound = numel(temp_hist);
end

% ensure that 75% of the pixels have been taken
% c = cumsum(temp_hist);
% idx = find(c>95,1);
norm_hist = temp_hist/max(temp_hist);
idx = find(norm_hist(hist_mode_loc:end)<0.05,1) + hist_mode_loc - 1;

upper_bound = max(idx, 18*hist_mode_loc);

% Compute the density metric
if upper_bound > numel(temp_hist)
    warning('upper bound set to end of list.');
    upper_bound = numel(temp_hist); 
end
density_metric = sum(temp_hist(lower_bound:upper_bound));

% Fit a line between the 80th and the 40th percentiles from the plot above
saturation1 = 3;
saturation2 = 42;
a = (95 - 40) / (saturation1 - saturation2);
b = 95 - a*saturation1;

% Compute gradient threshold
prct_value = round(a*density_metric + b);
if prct_value > 98
    prct_value = 98;
end
if prct_value < 25
    prct_value = 25;
end
% decrease or increase by a multiple of 5 percentile the manual input
prct_value = prct_value - greedy_step*manual_finetune; 
if prct_value > 100, prct_value = 100; end
if prct_value < 1, prct_value = 1; end
	
prct_value = prct_value/100;
threshold = percentile(S1,prct_value);


% Threshold the gradient image and perform some cleaning with morphological operations
S = S > threshold;
S = fill_holes(S, I, min_hole_size, max_hole_size, hole_min_perct_intensity, hole_max_perct_intensity, fill_holes_bool_oper);
S = imerode(S,strel('disk',1));  % removes the border pixels from the gradient
S = bwareaopen(S,min_cell_size,8);

end




