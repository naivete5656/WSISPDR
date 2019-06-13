% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.



% [seed_image, Highest_cell_number] =
%     fog_bank_perctile_geodist2(
%         grayscale_image,
%         foreground_mask,
%         mask_matrix,
%         min_peak_size,
%         min_object_size,
%         perc_binning,             <optional>
%

function [seed_image, nb_peaks] = fog_bank_perctile_geodist(grayscale_image, foreground_mask, mask_matrix, min_peak_size, min_object_size, fogbank_direction, perc_binning)


% get the image to be segmented
grayscale_image = single(grayscale_image);
[nb_rows, nb_cols] = size(grayscale_image);

if nargin == 5
  fogbank_direction = 1; % 1 is min to max
  perc_binning = 5;
end
if nargin == 6
  perc_binning = 5;
end

assert(islogical(foreground_mask), 'fog_bank_perctile_geodist:argCk','Invalid <foreground_mask>, Type');
assert(size(foreground_mask,1) == nb_rows && size(foreground_mask,2) == nb_cols, 'fog_bank_perctile_geodist:argCk','Invalid <foreground_mask>, wrong size');
assert(islogical(mask_matrix),'fog_bank_perctile_geodist:argCk','Invalid <mask_matrix> Type');
assert(size(mask_matrix,1) == nb_rows && size(mask_matrix,2) == nb_cols, 'fog_bank_perctile_geodist:argCk','Invalid <mask_matrix> wrong size');
assert(min_peak_size > 0, 'fog_bank_perctile_geodist:argCk','Invalid <min_peak_size>');
assert(min_object_size > 0, 'fog_bank_perctile_geodist:argCk','Invalid <min_object_size>');
assert(isnan(perc_binning) || (perc_binning >= 0 && perc_binning < 100), 'fog_bank_perctile_geodist:argCk','Invalid <percentile_binning>');

% transform background to nan and get minimum value on cell area
grayscale_image(~foreground_mask) = NaN;

if isnan(perc_binning) || perc_binning == 0
  Y = unique(grayscale_image(isfinite(grayscale_image)))';
else
  P_vec = (0:perc_binning:100)/100;
  Y = percentile(grayscale_image(:),P_vec);
end

Y = unique([0, Y]);
if fogbank_direction
  Y = sort(Y,'ascend');
else
  Y = sort(Y,'descend');
end

% based on the distance transform matrix, gradually drop a fog from the sky down to the ground passing through
% all the mountains in between and keeping them separated from one another.
%
% Find the first image where the fog will uncover some mountain peak that meet the minimum size threshold
% Initialize the image_b where the fog covers everything but the mountain_top

fog_level = 1;
nb_objects = 0;
while nb_objects == 0 && fog_level <= numel(Y)
  if fogbank_direction
    CC = bwconncomp(grayscale_image <= Y(fog_level) & mask_matrix, 8);
  else
    CC = bwconncomp(grayscale_image >= Y(fog_level) & mask_matrix, 8);
  end
  % determine the number of objects that meet the min peak size threshold
  k = 0;
  for i = 1:CC.NumObjects
    if numel(CC.PixelIdxList{i}) >= min_peak_size
      k = k + 1;
    end
  end
  nb_objects = k;
  fog_level = fog_level + 1;
end
% determine the type of matrix that is required to hold the data
dataType = get_min_required_datatype(nb_objects);
% init the seed image to hold the labeled cell bodies
seed_image = zeros(CC.ImageSize,dataType);
nb_peaks = 0;
% populate seed image with the objects that meet the min peak size threshold
for k = 1:CC.NumObjects
  if numel(CC.PixelIdxList{k}) >= min_peak_size
    nb_peaks = nb_peaks + 1;
    seed_image(CC.PixelIdxList{k}) = nb_peaks;
  end
end


% Start dropping the fog
for n = fog_level:numel(Y)
  
  %     imwrite(uint16(seed_image), ['D:\temp\FB_test\' sprintf('test_%04d.tif',n)]);
  
  % get the binary image containing the pixels that are to be assigned a label at this fog level
  if fogbank_direction
    image_b = grayscale_image <= Y(n) & mask_matrix;
  else
    image_b = grayscale_image >= Y(n) & mask_matrix;
  end
  
  
  % assign non zero pixels in image_b the label of the closest connected peak in seed_image
  seed_image = assign_nearest_connected_label(seed_image, image_b);
  
  % remove the pixels from image_b that have already been assigned a label
  image_b(seed_image>0) = 0;
  
  % find any new peaks (groups of pixels in image_b that have not been assigned a label that meet the min peak size threshold)
  CC = bwconncomp(image_b ,8);
  % work out if seed_image's type needs to be expanded to accommodate larger label values
  new_max_val = nb_peaks;
  for k = 1:CC.NumObjects
    if numel(CC.PixelIdxList{k}) >= min_peak_size
      new_max_val = new_max_val + 1;
    end
  end
  % if the required datatype has expanded, cast the seed image
  dataType = get_min_required_datatype(new_max_val);
  if ~strcmpi(dataType, class(seed_image))
    seed_image = cast(seed_image, dataType);
  end
  % add the new peaks into the seed image
  for k = 1:CC.NumObjects
    if numel(CC.PixelIdxList{k}) >= min_peak_size
      nb_peaks = nb_peaks + 1;
      seed_image(CC.PixelIdxList{k}) = nb_peaks;
    end
  end
end

% assign any un assigned pixels to the nearest body
seed_image = assign_nearest_connected_label(seed_image, foreground_mask);

% Scout all the pixels in the image looking for the non background pixels
indx = find(seed_image);
objects_size = zeros(nb_peaks, 1);
for i = 1:numel(indx)
  % Increment the size of the pixel
  objects_size(seed_image(indx(i))) = objects_size(seed_image(indx(i))) + 1;
end
% Delete cells with size less than threshold
[seed_image, nb_peaks] = check_cell_size(seed_image, nb_peaks, objects_size, min_object_size);

% Check for objects connectivity in the image after performing the separation by the sticking fog.
seed_image = check_body_connectivity(seed_image, nb_peaks);

end



function dataType = get_min_required_datatype(maxVal)
if maxVal <= intmax('uint8')
  dataType = 'uint8';
elseif maxVal <= intmax('uint16')
  dataType = 'uint16';
elseif maxVal <= intmax('uint32')
  dataType = 'uint32';
else
  dataType = 'double';
end
end


function [img, highest_cell_number] = check_cell_size(img, Highest_cell_number, cell_size, cell_size_threshold)

% Create a renumber_cells vector that contains the renumbering of the cells with size > min_size
renumber_cells = zeros(Highest_cell_number+1, 1);
highest_cell_number = 0;
for i = 1:Highest_cell_number
  % if cell i is a cell with size > min_size, give it a new number
  if cell_size(i) > cell_size_threshold
    highest_cell_number = highest_cell_number + 1;
    renumber_cells(i+1) = highest_cell_number;
  end
end

% Delete small cells
BW = img > 0;
img = renumber_cells(img+1);
% assign deleted pixels the label of the nearest connected body
img = assign_nearest_connected_label(img, BW);

end





