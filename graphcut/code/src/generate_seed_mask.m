% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


% image: the image to use in generating the seed mask
% img_filter: filter operation to apply to the image
% 	(None, Gradient, Entropy, Std)
% prctile_threshold: threshold defining foreground
% threshold_direction: wether above or below the threshold is considered foreground
% 	(<,>=)
% min_obj_size: minimum area in pixels of an object
% max_obj_size: maximum area in pixels of an object
% circularity_threshold: minimum circularity objects must have
% cluster_distance
% foreground_mask (optional)
% border_mask (optional)


function BW = generate_seed_mask(image, img_filter, percentile_thresholdL, threshold_operatorL, percentile_thresholdR, threshold_operatorR, min_obj_size, max_obj_size, circularity_threshold, cluster_distance, foreground_mask, border_mask)

if ~exist('foreground_mask','var')
  foreground_mask = ones(size(image));
end
if ~exist('img_filter','var')
  img_filter = 'none';
end


% input checking
assert(exist('image','var')>0, 'Image to generate seed from missing');
assert(exist('percentile_thresholdL','var')>0, 'Percentile threshold used to generate seed from missing');
assert(exist('percentile_thresholdR','var')>0, 'Percentile threshold used to generate seed from missing');
assert(exist('min_obj_size','var')>0, 'Minimum object size missing');
assert(exist('max_obj_size','var')>0, 'Maximum object size missing');

assert(percentile_thresholdL >= 0 && percentile_thresholdL <= 100);
assert(percentile_thresholdR >= 0 && percentile_thresholdR <= 100);
percentile_thresholdL = percentile_thresholdL./100;
percentile_thresholdR = percentile_thresholdR./100;
assert(size(image,1) == size(foreground_mask,1) && size(image,2) == size(foreground_mask,2), 'Seed mask image must be the same size as the foreground mask');


image = double(image);
% modify the foreground mask to account for the border mask
if exist('border_mask', 'var') && ~isempty(border_mask)
  foreground_mask = foreground_mask & ~border_mask;
end

img_filter = lower(regexprep(img_filter, '\W', ''));
switch img_filter
  case 'gradient'
    image = imgradient(image, 'Sobel');
  case 'entropy'
    image = entropyfilt(image,true(5,5));
  case 'std'
    image = stdfilt(image,true(5,5));
  otherwise
    % do nothing
end

% threshold the image
% P = percentile(image(:), [prctile_thresholdL,prctile_thresholdR]);
P = percentile(image(foreground_mask), [percentile_thresholdL,percentile_thresholdR]);
switch threshold_operatorL
  case '>'
    BW_L = image > P(1);
  case '<'
    BW_L = image < P(1);
  case '>='
    BW_L = image >= P(1);
  case '<='
    BW_L = image <= P(1);
  otherwise
    error('Invalid threshold operator');
end
switch threshold_operatorR
  case '>'
    BW_R = image > P(2);
  case '<'
    BW_R = image < P(2);
  case '>='
    BW_R = image >= P(2);
  case '<='
    BW_R = image <= P(2);
  otherwise
    error('Invalid threshold operator');
end

BW = BW_L & BW_R;

% remove any pixels from the seed mask that are not part of the overall foreground
BW = BW & foreground_mask;

% fill in the holes that are smaller than twice the minimum object size
BW = fill_holes(BW, [], min_obj_size*2);
% remove objects that are smaller than the minimum object size
BW = bwareaopen(BW, min_obj_size);

% remove the objects in BW above seed_max_object_size
CC = bwconncomp(BW);
for ii = 1:CC.NumObjects
  if numel(CC.PixelIdxList{ii}) > max_obj_size
    BW(CC.PixelIdxList{ii}) = 0; % remove the object by setting its pixels to zero
  end
end


if exist('circularity_threshold','var')
  % circularity check, only keeps cells below the circularity threshold
  BW = filter_by_circularity(BW, circularity_threshold);
end

if exist('cluster_distance','var')
  % cluster seed objects
  BW = cluster_objects(BW, cluster_distance, foreground_mask);
end


end