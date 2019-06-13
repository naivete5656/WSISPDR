% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


% apply a morphological operaton to an image and mask
function BW = morphOp(I, BW, op_str, radius, border_mask)
if nargin == 5
    use_border_flag = true;
else
    use_border_flag = false;
    border_mask = [];
end

if radius == 0
    return; % radius of 0 will have no affect
end

border_mask = logical(border_mask);


op_str = lower(regexprep(op_str, '\W', ''));
switch op_str
    case 'dilate'
        if use_border_flag
            BW = geodesic_imdilate(BW, ~border_mask, radius);
        else
            BW = imdilate(BW, strel('disk', radius));
        end
    case 'erode'
        BW = imerode(BW, strel('disk', radius));
    case 'close'
        if use_border_flag
            BW = geodesic_imclose(BW, ~border_mask, radius);
        else
            BW = imclose(BW, strel('disk', radius));
        end
    case 'open'
        if use_border_flag
            BW = geodesic_imopen(BW, ~border_mask, radius);
        else
            BW = imopen(BW, strel('disk', radius));
        end
        
    case 'iterativegraydilate'
        BW = iterative_geodesic_gray_dilate(I, BW, ~border_mask, radius, 0.5);
end

end