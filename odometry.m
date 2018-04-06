%% Purge
close all; clear all; clc

%% Assign Path based on OS
if ispc
    PARENTPATH = 'D:\Users\alexg636\Downloads\Oxford_dataset\Oxford_dataset\';
    IMGDIR = strcat(PARENTPATH, 'stereo\centre\');
    MODELSDIR = strcat(PARENTPATH, 'model\');
elseif ismac
    PARENTPATH = '/Users/alexgeorge/Downloads/Oxford_dataset/';
    IMGDIR = strcat(PARENTPATH, 'stereo/centre/');
    MODELSDIR = strcat(PARENTPATH, 'model/');
end

imgs = dir(strcat(IMGDIR, '*.png'));
img_name = {imgs.name};

%% Setup Video
% outputVid = VideoWriter(strcat(pwd, '/Oxford_Trajectory2.avi'));
% outputVid.FrameRate = 25;
% open(outputVid)
%% Camera Setup
[fx, fy, cx, cy, G_camera_image, LUT] = ReadCameraModel(IMGDIR, MODELSDIR);

% Orient initial camera center.
camCentA = [];
camCentB = [];
camCent_x = 0;
camCent_y = 0;
camCent_z = 0;

McamCent_x = 0;
McamCent_y = 0;
McamCent_z = 0;

% Lower triangular intrinsic camera matrix
K = [fx 0  0;
     0 fy 0;
     cx cy 1];

camIntrinsic = cameraIntrinsics([fx fy], [cx cy], size(strcat(IMGDIR, img_name{1})));

% https://en.wikipedia.org/wiki/Essential_matrix
W = [0 -1 0;
     1 0 0;
     0 0 1];
 
invW = inv(W);
 
Z = [0 1 0;
     -1 0 0;
     0 0 0];
 
figure('units', 'normalized', 'outerposition', [0 0 1 1])

Dist = 0;
%% Loop Beginning
% Begin at frame 20 to avoid glare at beginning
frame = 20;
ind = 2;

RT = eye(4);
RT2 = eye(4);

for i = frame:length(img_name)
    img_path_i = strcat(IMGDIR, img_name{i-1});
    img_path_f = strcat(IMGDIR, img_name{i});

    %% Image Processing

    % Demosaic --> Undistort
    img_ic = UndistortImage(demosaic(imread(img_path_i),'gbrg'), LUT);
    img_fc = UndistortImage(demosaic(imread(img_path_f),'gbrg'), LUT);
    
    % Convert to grayscale
    img_i = rgb2gray(img_ic);
    img_f = rgb2gray(img_fc);
    
    %% Corresponding Points

    % Detect corresponding points between subsequent images.
    pts_i = detectSURFFeatures(img_i);
    pts_f = detectSURFFeatures(img_f);

    % Extract features.
    [f_i, vpts_i] = extractFeatures(img_i, pts_i);
    [f_f, vpts_f] = extractFeatures(img_f, pts_f);

    % Match corresponding points.
    indexPairs = matchFeatures(f_i, f_f, 'Unique', true);
    matchedPoints1 = vpts_i(indexPairs(:, 1));
    matchedPoints2 = vpts_f(indexPairs(:, 2));

    %% User-defined Trajectory
    
    %  Fundamental Matrix using RANSAC
    [FundMtx, inlierPts1, inlierPts2] = ransac_FMtx(...
                                                matchedPoints1.Location,... 
                                                matchedPoints2.Location,...
                                                2000,...
                                                8,...
                                                1e-4);
        
    % Essential Matrix
    E_loc = K*FundMtx*K';

    % Decompose Essential Matrix
    [Rot, Tsn] = EMtx2RT(E_loc, W, Z);
    
    % Select Rotation|Translation based on positive depth
    [RotT, TraT] = camPosDepth(Rot, Tsn, K, inlierPts1, inlierPts2);
    RT2 = RT2*vertcat(horzcat(RotT',-TraT'), [0 0 0 1]);
    
    % Transformed points
    camCent_x(ind) = RT2(1,4);
    camCent_y(ind) = RT2(2,4);
    camCent_z(ind) = RT2(3,4);
    
    %% MATLAB-based Trajectory

    % Fundamental Matrix using RANSAC
    [M, inliersIndex] = estimateFundamentalMatrix(...
                        matchedPoints1,...
                        matchedPoints2,... 
                        'Method', 'RANSAC',... 
                        'NumTrials', 2000,... 
                        'DistanceThreshold', 1e-4);

    % Rotation and Translation
    mat_inliers1 = matchedPoints1.Location(inliersIndex, :);
    mat_inliers2 = matchedPoints2.Location(inliersIndex, :);
    [RotM, TraM] = relativeCameraPose(M, camIntrinsic,... 
                   mat_inliers1,... 
                   mat_inliers2);      

    % Transformed points
    RT = RT*vertcat(horzcat(RotM',TraM'), [0 0 0 1]);
    McamCent_x(ind) = RT(1,4);
    McamCent_y(ind) = RT(2,4);
    McamCent_z(ind) = RT(3,4);
    
    %% Euclidean distance between points
    VV = [McamCent_x(ind), McamCent_z(ind)] - ...
        [camCent_x(ind), camCent_z(ind)];
    Diff = sqrt(VV*VV');
    Dist = Dist + Diff;
          
        %% Plot both
        subplot(3,3,1)
        imshow(img_ic);
        title(sprintf('Oxford Drive Videostream - %d/3873', frame+ind-2))
        
        subplot(3,3,4);
        showMatchedFeatures(img_i, img_f, mat_inliers2, mat_inliers2);
        title(sprintf('Matlab-based - %d inliers', size(mat_inliers1, 1)));
        
        subplot(3,3,7)
        UB=showMatchedFeatures(img_i, img_f, inlierPts1, inlierPts2);
        title(sprintf('User-based - %d inliers', size(inlierPts1, 1)));
        
        subplot(3,3,[2,3,5,6])
        hold on
        title('Approximated Vehicle Trajectory', 'Fontsize', 24)
        plot(McamCent_x(ind-1:ind), McamCent_z(ind-1:ind), 'r')
        plot(camCent_x(ind-1:ind), camCent_z(ind-1:ind), 'b')
        daspect([2 2 2]); pbaspect([2 2 2])
        lgd = legend('Matlab-based', 'User-based');
        lgd.FontSize = 16;
        xlabel(sprintf('Current Drift: %f units', round(Diff, 3)), 'Fontsize', 16);
        hold off
        
        subplot(3,3,[8,9])
        hold on
        grid on
        bar(frame+ind-2, Dist)
        xlabel(sprintf('Accumulated Drift: %f units', round(Dist, 3)), 'Fontsize', 16);
        ylabel('Drift (units)') 
        hold off
        drawnow
        
%         movieFrame = getframe(gcf);
%         writeVideo(outputVid, movieFrame)
        
        
    ind = ind+1;
end

% close(outputVid)



