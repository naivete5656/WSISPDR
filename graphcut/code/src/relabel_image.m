% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

function [segmented_image, nb_cell] = relabel_image(segmented_image)

curclass = class(segmented_image);
% Create a renumber_cells vector that contains the renumbering of the cells in the labeled mask
nb_cell = max(segmented_image(:));
renumber_cells = zeros(nb_cell, 1);
[m, n] = size(segmented_image);

% Get unique cell ID
u = unique(segmented_image(:));
if u(1) == 0, u(1) = []; end % delete background pixel

for i = 1:length(u), renumber_cells(u(i)) = i; end
renumber_cells = [0;renumber_cells]; % Account for background
renumber_cells = cast(renumber_cells, curclass);

% Renumber image
segmented_image = renumber_cells(segmented_image+1);
segmented_image = reshape(segmented_image, m, n);

% Nb of cells
nb_cell = length(u);


