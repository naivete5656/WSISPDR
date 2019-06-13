% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

% This function will plot the segmented image with each object in the object assigned with different color
function segplot(S, txt, colors_vector, shuffle)

% Compute the number of objects in the image
N = zeros(max(S(:))+1,1); N(S(:)+1) = 1; N = find(N)-1; N(1) = []; % Find Unique values (faster thena the fuinciton'unique')
max_objects = N(end);
nb_objects = length(N); % number of objects without background

% If user specified a color use it, otherwise create new one with random shuffling
if nargin < 3 || isempty(colors_vector), colors_vector = jet(double(max_objects)); end
if nargin < 4 || shuffle == 1, colors_vector = colors_vector(randperm(max_objects),:); end
if nargin < 2, txt = 1; end

% Make colored image
image_RGB = label2rgb(S, colors_vector, 'k');

% Plot the segmented_image
image(image_RGB), title(['Number of objects in image = ' num2str(nb_objects)]); hold on

% Get text location Ind
Ind = zeros(max_objects,2);
[rows, cols] = size(S);
for j = 1:cols
  for i = 1:rows
    if ~S(i,j) || Ind(S(i,j),1), continue, end
    Ind(S(i,j),1) = i;
    Ind(S(i,j),2) = j;
  end
end
Ind = Ind(Ind(:,1)>0,:); % remove un-assigned objects in the image. if numbers are not sequential.

% Place the number of the object in the segmented_image
if txt == 1
  for i = 1:nb_objects
    text(Ind(i,2), Ind(i,1), num2str(N(i)), 'fontsize', 6,'FontWeight', 'bold', 'Margin', .1, 'color', 'k', 'BackgroundColor', 'w')
  end
end

