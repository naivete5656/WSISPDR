% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


function BW = cluster_objects(BW, cluster_distance, valid_traversal_mask)

if nargin == 2
  valid_traversal_mask = true(size(BW));
end

% label the connected component objects to cluster
[BW, nb_obj] = bwlabel(BW);

% allocate space for the cluster labels
cluster_labels = zeros(nb_obj,1);

% generate the centroids image to use as the seeds for geodesic distance
stats = regionprops(BW, 'Centroid', 'BoundingBox');

% round the centroids to the nearest integer
centroids = zeros(size(stats,1),2);
for i = 1:numel(stats)
  centroids(i,:) = stats(i).Centroid;
end
centroids = round(centroids);

% compute the geodesic distance between object centroids to cluster
BWd = bwdistgeodesic(valid_traversal_mask, centroids(:,1), centroids(:,2));

% generate clusters by thresholding the distance transform image
clusters = bwlabel(BWd <= cluster_distance/2);

% assign clusters labels
for i = 1:size(centroids,1)
  % obtain the centroid location
  cent = [centroids(i,2), centroids(i,1)];
  % look up that centroid's cluster id
  cluster_label = clusters(cent(1), cent(2));
  
  % assign that object to the cluster
  cluster_labels(i) = cluster_label;
end

cluster_labels = [0;cluster_labels];
% relabel the image
BW = cluster_labels(BW+1);


end