% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


function S = fill_holes_between(S, lower_bound, upper_bound)

S = logical(S);

if isinf(upper_bound)
  % the user has selected no upper limit (inf) for the fill holes size
  
  % fill holes that are larger than lower_bound but smaller than the max hole size(the background)
  BW = ~S;
  CC = bwconncomp(BW,4);
  szs = zeros(numel(CC.PixelIdxList),1);
  for i = 1:numel(CC.PixelIdxList)
    szs(i,1) = numel(CC.PixelIdxList{i});
  end
  upper_bound = max(szs(:)) - 1;
  if isempty(upper_bound)
    upper_bound = 0;
  end
end

% BWu holds the mask of pixels where the holes larger than upper_bound have been filled
BWu = S;
BWu(~bwareaopen(~S, upper_bound, 4)) = 1;

% BWl holds the mask of pixels where the holes smaller than lower_bound have been filled
BWl = S;
BWl(~bwareaopen(~S, lower_bound, 4)) = 1;

% remove the hole pixels from S that are in either the upper or lower mask
S(xor(BWu, BWl)) = 1;

end