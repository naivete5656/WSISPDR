% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.



function [marker_matrix, dist_mat] = labeled_geodesic_dist(marker_matrix, mask_matrix)

[m,n] = size(marker_matrix);

assert(islogical(mask_matrix));
assert(size(mask_matrix,1) == m);
assert(size(mask_matrix,2) == n);

dist_mat = inf(m,n);
dist_mat(marker_matrix>0) = 0;
dist_mat(~mask_matrix) = NaN;

init = m;
indx = 0;
loc_vec = zeros(init,1);
lab_vec = zeros(init,1);

% find edge pixels
[ii, jj] = find(marker_matrix);
for k = 1:numel(ii)
  i = ii(k);
  j = jj(k);
  if i == 1 || j == 1 || i == m || j == n || ...
      ~marker_matrix(i-1,j-1) || ...
      ~marker_matrix(i,j-1) || ...
      ~marker_matrix(i+1,j-1) || ...
      ~marker_matrix(i-1,j) || ...
      ~marker_matrix(i+1,j) || ...
      ~marker_matrix(i-1,j+1) || ...
      ~marker_matrix(i,j+1) || ...
      ~marker_matrix(i+1,j+1)
    
    push(i,j);
  end
end

edge_pixels = [loc_vec(1:indx) lab_vec(1:indx)];
indx = 0;

iteration_count = 0;

while size(edge_pixels,1) > 0
  iteration_count = iteration_count + 1;
  
  for k = 1:size(edge_pixels,1)
    % for each pixel that has a label, find all its neighbors that dont have a label and are valid for traversal
    i = edge_pixels(k,1);
    j = edge_pixels(k,2);
    label = marker_matrix(i,j);
    % check 4 connected nhood
    ii = i; jj = j-1; % left
    if jj > 0 && marker_matrix(ii,jj) == 0 &&  mask_matrix(ii,jj)
      marker_matrix(ii,jj) = label;
      push(ii,jj);
      edge_pixels(k,1) = -1;
    end
    ii = i - 1; jj = j; % top
    if ii > 0 && marker_matrix(ii,jj) == 0 &&  mask_matrix(ii,jj)
      marker_matrix(ii,jj) = label;
      push(ii,jj);
      edge_pixels(k,1) = -1;
    end
    ii = i + 1; jj = j; % bottom
    if ii <= m && marker_matrix(ii,jj) == 0 &&  mask_matrix(ii,jj)
      marker_matrix(ii,jj) = label;
      push(ii,jj);
      edge_pixels(k,1) = -1;
    end
    ii = i; jj = j+1; % right
    if jj <= n && marker_matrix(ii,jj) == 0 &&  mask_matrix(ii,jj)
      marker_matrix(ii,jj) = label;
      push(ii,jj);
      edge_pixels(k,1) = -1;
    end
  end
  
  % update the new edge pixels
  edge_pixels = [loc_vec(1:indx) lab_vec(1:indx)];
  indx = 0;
  % update the distance matrix
  for k = 1:size(edge_pixels,1)
    dist_mat(edge_pixels(k,1),edge_pixels(k,2)) = iteration_count;
  end
  
  
  temp = (edge_pixels(:,2)-1).*m + edge_pixels(:,1);
  [~, ind_vec] = sort(temp, 'ascend');
  edge_pixels = edge_pixels(ind_vec,:);
end

  function push(val1, val2)
    if indx >= numel(loc_vec)
      loc_vec = [loc_vec; zeros(init,1)];
      lab_vec = [lab_vec; zeros(init,1)];
      init = numel(loc_vec);
    end
    indx = indx + 1;
    loc_vec(indx) = val1;
    lab_vec(indx) = val2;
  end
end


