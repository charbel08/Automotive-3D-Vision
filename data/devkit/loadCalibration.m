function [K, P_left, P_right] = loadCalibration(calibfile)
% LOADCALIBRATION provides all needed coordinate system transformations
% returns the pre-computed velodyne to cam (gray and color) projection

% get the camera intrinsic and extrinsic calibration
calib = loadCalib(calibfile);

% calibration matrix after rectification (equal for all cameras)
K = calib.P_rect{1}(:,1:3);
P_left = calib.P_rect{3};
P_right = calib.P_rect{4};
end


function calib = loadCalib(calibfile)
    
calib = getcam(calibfile);

end

function [calib] = getcam(calibfile, f)

% f is the image resize factor

if nargin < 3
    f = 1;
end;
   fid = fopen(calibfile);
  % load 3x4 projection matrix
    C = textscan(fid,'%s %f %f %f %f %f %f %f %f %f %f %f %f',4);
  calib = [];
  for j = 0 : 3
     P = [];
     for i=0:11
       P(floor(i/4)+1,mod(i,4)+1) = C{i+2}(j+1);
     end
     calib.P_rect{j+1} = P;
  end;
  C = textscan(fid,'%s %f %f %f %f %f %f %f %f %f',1);
  
  % load R_rect
%   C = textscan(fid,'%s %f %f %f %f %f %f %f %f %f',1);
  
  % load velo_to_cam
%   C = textscan(fid,'%s %f %f %f %f %f %f %f %f %f %f %f %f');
  
  fclose(fid);
end
