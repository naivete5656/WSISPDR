% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.



function BW = generate_border_mask(image, img_filter, percentile_threshold, threshold_direction, border_break_holes_flag, border_thin_mask_flag, foreground_mask)

if ~exist('foreground_mask','var')
  foreground_mask = ones(size(image));
end
if ~exist('img_filter','var')
  img_filter = 'none';
end
if ~exist('border_break_holes_flag','var')
  border_break_holes_flag = false;
end
if ~exist('border_thin_mask_flag','var')
  border_thin_mask_flag = false;
end

% input checking
assert(exist('image','var')>0, 'Image to generate seed from missing');
assert(exist('percentile_threshold','var')>0, 'Percentile threshold used to generate seed from missing');

assert(percentile_threshold >= 0 && percentile_threshold <= 100);
percentile_threshold = percentile_threshold./100;
assert(size(image,1) == size(foreground_mask,1) && size(image,2) == size(foreground_mask,2), 'border mask image must be the same size as the foreground mask');

image = double(image);

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
P = percentile(image(:), percentile_threshold);
switch threshold_direction
  case '>'
    BW = image > P(1);
  case '<'
    BW = image < P(1);
  case '>='
    BW = image >= P(1);
  case '<='
    BW = image <= P(1);
  otherwise
    error('invalid threshold operator');
end

% remove any pixels from the seed mask that are not part of the overall foreground
BW = BW & foreground_mask;

% morphological cleanup
BW = imdilate(BW,strel('disk',1));
BW = bwmorph(BW, 'bridge');
BW = bwmorph(BW, 'thin',1);
BW = bwmorph(BW, 'diag');

BW = fill_holes(BW, [], 10); % fill holes smaller han 10 pixels

if border_thin_mask_flag
  BW = bwmorph(BW, 'thin',inf);
end
BW = bwmorph(BW, 'diag');

if border_break_holes_flag
  % break the holes so they are no longer fillable
  BW = break_holes(image, foreground_mask, BW);
end

end