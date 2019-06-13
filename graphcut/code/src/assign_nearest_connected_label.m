% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


% assignes all of the true pixels in mask_matrix the value of the nearest connected pixel in marker_matrix
function marker_matrix = assign_nearest_connected_label(marker_matrix, mask_matrix)

marker_matrix = int32(marker_matrix);

try
  % labeled_geodesic_dist_mex
  % check for the mex file binaries that are required to run the mex file
  mex_file_name = ['labeled_geodesic_dist_mex.' mexext];
  if ~exist(mex_file_name, 'file') && exist('labeled_geodesic_dist_mex.c','file')
    % check the compiler configuration, do not compile unless there is a selected C Compiler
    if ~isempty(mex.getCompilerConfigurations('C','Selected'))
      eval(['mex ' which('labeled_geodesic_dist_mex.c')]);
    end
  end
  % if the required mex file exists
  if exist(mex_file_name, 'file')
    % call the mex version and return the result
    [marker_matrix, ~] = labeled_geodesic_dist_mex(marker_matrix, mask_matrix);
    return;
  end
catch
  % warn the user if the mex compiliation was attempted and failed
  warning('mex function failed, using Matlab version');
end

% if the mex cannot be found/compiled call the slower non-mex verion
[marker_matrix, ~] = labeled_geodesic_dist(marker_matrix, mask_matrix);

end
