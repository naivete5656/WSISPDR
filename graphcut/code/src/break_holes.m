% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


% I1 is the double precision image holding the pixel wights
% BW is the logical mask containing the holes to be broken
function BW = break_holes(I1, S, BW)

I1 = double(I1);

% create a mask with the holes filled
filled_BW = imfill(BW, 'holes');
% remove the background from the filled holes mask
filled_BW(~S) = 0;
% create the xor mask between the holes to be broken mask and the filled version
xor_img = xor(BW, filled_BW);
% only keep object larger than 3 pixels
xor_img = bwareaopen(xor_img,3);

% if no holes needing to be broken exist, return
if max(xor_img(:)) == 0, return, end

I1 = I1.^2; % weight the grayscale pixels
I1(xor_img) = 0;

% the loop handles nested holes
iter = 1;
while iter<100 % provide an absolute cuttoff to prevent this from running forever
  % compute the centroids
  stats = regionprops(xor_img, 'Centroid');
  % round the centroid values to the nearest integer
  c = zeros(numel(stats),2);
  for i = 1:numel(stats)
    c(i,:) = round(stats(i).Centroid);
  end
  
  % compute the grayscale distance between the background and all locations
  gd2 = graydist(I1, ~filled_BW);
  % compute the grayscale distance between the centroids and all locations
  gd1 = graydist(I1, c(:,1),c(:,2));
  % find the minimal basin connecting the hole centroids to the background
  break_pixels = imregionalmin(gd1 + gd2)>0;
  % connect the basins to 8 connectivity
  break_pixels = bwmorph(break_pixels, 'diag');
  
  % break the hole by setting the mask to 0
  BW(break_pixels) = 0;
  
  % perform filling to recompute the hole mask taking into account the newly broken holes
  filled_BW = imfill(BW, 'holes');
  filled_BW(~S) = 0;
  xor_img = xor(BW, filled_BW);
  xor_img = bwareaopen(xor_img,3);
  
  % if there are no more holes to break, exit the loop
  if max(xor_img(:)) == 0, break, end
  iter = iter + 1;
end




