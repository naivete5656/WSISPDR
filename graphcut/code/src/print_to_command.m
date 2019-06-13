% NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


function log_file_path = print_to_command(value, tgt_dir)

switch nargin
  case 1
    tgt_dir = '';
    use_log_file_flag = false;
  case 2
    if ~ischar(tgt_dir)
      use_log_file_flag = false;
    else
      use_log_file_flag = true;
    end
    % this is the correct number of arguments
  otherwise
    return;
end

if ~isa(value,'numeric') && ~isa(value,'char')
  error('print_to_command:argChk','Able to print numeric and char types');
end

newline_char = '\n';

date_vector = round(clock());
time_vector = date_vector(4:end);
if use_log_file_flag
  [~,~,ext] = fileparts(tgt_dir);
  if ~isempty(ext)
    log_file_path = tgt_dir;
  else
    log_file_path = [tgt_dir filesep getenv('username') '_log' sprintf('_%04d%02d%02d',date_vector(1),date_vector(2),date_vector(3)) '.log'];
  end
  log_fileID = fopen(log_file_path,'a');
end
output_padding = '          ';

output_string = sprintf('<%02d:%02d:%02d>',time_vector(1),time_vector(2),time_vector(3));

if isa(value, 'char') && isempty(value)
  % simply print a clear clean line
  fprintf(1, newline_char);
  if use_log_file_flag
    fprintf(log_fileID, newline_char);
  end
else
  if ~isempty(value)
    if ~isa(value, 'char')
      value = num2str(value);
    end
    [m,~] = size(value);
    fprintf(1, [output_string ' %s' newline_char], value(1,:));
    if use_log_file_flag
      fprintf(log_fileID, [output_string ' %s' newline_char], value(1,:));
    end
    for i = 2:m
      fprintf(1, [output_padding ' %s' newline_char], value(i,:));
      if use_log_file_flag
        fprintf(log_fileID, [output_padding ' %s' newline_char], value(i,:));
      end
    end
  end
end

if use_log_file_flag
  fclose(log_fileID);
end

end



function [cur_path] = validate_filepath(cur_path)
if nargin ~= 1, return, end
if ~isa(cur_path, 'char')
  error('validate_filepath:argChk','invalid input type');
end

[status,message] = fileattrib(cur_path);
if status == 0
  error('validate_filepath:notFoundInPath',...
    'No such file or directory: \"%s\"',cur_path);
else
  cur_path = message.Name;
  if message.directory == 0
    % the path represents a file so this is valid
    % So do nothing
  else
    % the path represents a directory
    if cur_path(end) ~= filesep
      cur_path = [cur_path filesep];
    end
  end
end

end