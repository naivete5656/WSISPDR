% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


function [image_RGB, colors_vector] = superimpose_colormap_contour(I1, segmented_image, cmap, contour_color, display_raw_image, display_contour, adjust_contrast, colors_vector)

% Compute the number of objects in the image
objects_numbers = unique(segmented_image(:)); % including the background number which is 1
max_objects = double(objects_numbers(end));
if objects_numbers(1) == 0, objects_numbers(1) = []; end    % without background
nb_objects = length(objects_numbers);

% colors_vector = [0 0 0];
% colors_vector = [1 1 1];
if nargin < 8
  colors_vector = jet(max_objects);
  colors_vector = colors_vector(randperm(max_objects),:);
end

if max_objects == 1
  
  if(strcmp(contour_color,'White') || strcmp(contour_color,'white') || strcmp(contour_color,'W') || strcmp(contour_color,'w'))
    colors_vector = [1 1 1];
  elseif(strcmp(contour_color,'Black') || strcmp(contour_color,'black') || strcmp(contour_color,'K') || strcmp(contour_color,'k'))
    colors_vector = [0 0 0];
  elseif(strcmp(contour_color,'Green') || strcmp(contour_color,'green') || strcmp(contour_color,'G') || strcmp(contour_color,'g'))
    colors_vector = [0 1 0];
  elseif(strcmp(contour_color,'Blue') || strcmp(contour_color,'blue') || strcmp(contour_color,'B') || strcmp(contour_color,'b'))
    colors_vector = [0 0 1];
  else % red
    colors_vector = [1 0 0];
  end
  
end

[number_rows, number_columns] = size(I1);

if (display_contour)
  % Find the edges of segmentation and tracking of the image
  [edge_image, ~] = find_edges(segmented_image);
  
  % Dilate the edge_image to thicken the contour plot
  edge_image = uint32(imdilate(edge_image, strel('square', 1))); edge_image = edge_image(:);
  if size(colors_vector,1) == 1
    edge_image = uint8(edge_image > 0); % with colors vector only being one element long, the edges must be binary
  end
else
  edge_image = segmented_image;
  edge_image = uint32(edge_image);
  if size(colors_vector,1) == 1
    edge_image = uint8(edge_image > 0); % with colors vector only being one element long, the edges must be binary
  end
end


if display_raw_image
  
  % if the image is normalized 0-1 rescale it so that the cast to uint16 does not result in a binary image
  if max(I1(:)) <= 1
    I1 = single(I1);
    I1 = I1 - min(I1(:));
    I1 = I1./max(I1(:));
    I1 = I1.*65000;
    I1 = uint16(I1);
  end
  
  I1(isnan(I1)) = 0;
  
  % Adjust contrast
  if adjust_contrast, I1 = imadjust(uint16(I1)); end
  
  % Convert the matrix to a scaled image with intensities between 0 and 1
  I1 = mat2gray(I1);
  I1 = I1(:);
  
  I1 = uint32(round(I1.*(size(cmap,1)-1)));
  image_RGB = cmap(I1+1,:);
else
  image_RGB = zeros(numel(I1),3, 'single');
end


% Fuse the two images
% image_RGB = [I1, I1, I1];
image_RGB(edge_image>0,:) = colors_vector(nonzeros(edge_image),:);
image_RGB = reshape(image_RGB, [number_rows number_columns 3]);
end




