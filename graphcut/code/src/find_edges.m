% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


function [edge_image, text_location, perimeter] = find_edges(segmented_image)

% Compute the image dimensions
[nb_rows, nb_columns] = size(segmented_image);

% highest_number in the image
highest_number = max(segmented_image(:));

% Initialize the output image that contains only the edge pixels of the cells
edge_image = zeros(nb_rows, nb_columns, 'uint16');

% Initialize the perimeter vector that contains the number of pixel on the perimeter of each cell in the image
perimeter = zeros(highest_number, 1);

% Initialize the matrix text_location that contains the coordinates X and Y of the a pixel that belongs to the
% edge of a cell
text_location = zeros(highest_number, 2);

% Initialize the binary vector "first_occurance" that contains 0 if the cell was already encountered 1 if it's
% the first time we encounter this cell
first_occurance = ones(highest_number, 1);

% Scout all the pixels in segmented_image looking for the edge ones. An edge pixel will have at least one of it's
% eight neighbors that haas a different value than the pixel itself
m = 1; % index for the lines of the text_location matrix
for j = 1:nb_columns
  for i = 1:nb_rows
    
    % if pixel(i,j) is a background pixel: continue
    if segmented_image(i,j) == 0, continue, end
    
    % if pixel(i,j) is on the left border of the image, or if pixel(i,j) is on the right border of the image,
    % or if pixel(i,j) is on the top border of the image or if pixel(i,j) is on the top border of the image,
    % or if one of the pixels around pixel (i,j) ==> pixel (i,j) is an edge pixel
    if j == 1 || j == nb_columns || i == 1 || i == nb_rows || ...
        segmented_image(i-1,j-1) ~= segmented_image(i,j) || ...
        segmented_image(i-1,j) ~= segmented_image(i,j) || ...
        segmented_image(i-1,j+1) ~= segmented_image(i,j) || ...
        segmented_image(i,j-1) ~= segmented_image(i,j) || ...
        segmented_image(i,j+1) ~= segmented_image(i,j) || ...
        segmented_image(i+1,j-1) ~= segmented_image(i,j) || ...
        segmented_image(i+1,j) ~= segmented_image(i,j) || ...
        segmented_image(i+1,j+1) ~= segmented_image(i,j)
      
      % Assign pixel (i,j) as an edge pixel
      edge_image(i,j) = segmented_image(i,j);
      
      % Increase the size of the perimeter for cell(i,j)
      perimeter(edge_image(i,j)) = perimeter(edge_image(i,j)) + 1;
      
      % if it is the first time we encounter that cell, memorise the pixel location
      if first_occurance(edge_image(i,j)) > 0
        % Memorise the coordinates for that cell
        text_location(m, 1) = j;
        text_location(m, 2) = i;
        m = m + 1;
        % Set the first_occurance to false
        first_occurance(edge_image(i,j)) = 0;
      end
    end
  end
end