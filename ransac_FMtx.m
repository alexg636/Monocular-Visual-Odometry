function [ransFMtx, inliers1, inliers2] = ransac_FMtx(matchedPts1, matchedPts2, iterations, numPts, errThresh)
    % https://en.wikipedia.org/wiki/Random_sample_consensus#Matlab_implementation
     
    % Empty matrix placeholder
    ransFMtx = [];
    
    % Initialize number of inliers
    maxInliers = 0;
     
    % Add one to end of vectors
    Pts1 = horzcat(matchedPts1, ones(size(matchedPts1,1), 1));
    Pts1 = Pts1';
    Pts2 = horzcat(matchedPts2, ones(size(matchedPts2,1), 1));
     
    % Run specified number of iterations
    for i = 1:iterations
        index = randi(size(matchedPts1,1),[numPts,1]);
         
        % Estimate normalized fundamental matrix
        roughFMtx = norm_fMtx(matchedPts1(index,:),...
                              matchedPts2(index,:));
                                
        % Find inliers within an error's distance                         
        err = sum((Pts2.*(roughFMtx*Pts1)'),2);
        currInliers = size(find(abs(err) <= errThresh) , 1);
        
        % Update number of inliers
        if (currInliers > maxInliers)
           ransFMtx = roughFMtx; 
           maxInliers = currInliers;
        end    
    end
    
    % Return indices of inliers
    err = sum((Pts2.*(ransFMtx*Pts1)'),2);
    [Y,I]  = sort(abs(err),'ascend');
    
    % Return 10% of all inlier points
    inliers1 = matchedPts1(I(1:size(I)*0.10),:);
    inliers2 = matchedPts2(I(1:size(I)*0.10),:);

    %% Normalized Fundamental Matrix 
    % https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)
     
    function norm_FMtx = norm_fMtx(matchedPtsA, matchedPtsB)
 
    %% Centroid values for image B
    
    % xbar
    meanvals_A = mean(matchedPtsA);
    
    % x-xbar
    offsetA = matchedPtsA - repmat(meanvals_A, [size(matchedPtsA,1), 1]);
    
    % variance and stddev
    var_a = var(offsetA);
    sd_a = sqrt(var_a);
    
    K_A = [1/sd_a(1), 0,0;
           0,1/sd_a(2), 0;
           0,0,1]...
           *...
           [1,0,-meanvals_A(1);
           0,1,-meanvals_A(2);
           0,0,1];
       
       
    norm_A = K_A*horzcat(matchedPtsA, ones(size(matchedPtsA,1), 1))';
                    
    norm_A = norm_A';
     
    %% Centroid values for image B

    % xbar
    meanvals_B = mean(matchedPtsB);
    
    % x-xbar
    offsetB = matchedPtsB - repmat(meanvals_B, [size(matchedPtsB,1), 1]);
    
    % variance and stddev
    var_b = var(offsetB);
    sd_b = sqrt(var_b);
    
    K_B = [1/sd_b(1), 0,0;
           0,1/sd_b(2), 0;
           0,0,1]...
           *...
           [1,0,-meanvals_B(1);
           0,1,-meanvals_B(2);
           0,0,1];
       
       
    norm_B = K_B*horzcat(matchedPtsB, ones(size(matchedPtsB,1), 1))';
                    
    norm_B = norm_B';
    
    % Multiple new scaled matrices
    u = norm_A .* repmat(norm_B(:,1),[1,3]);
    v = norm_A .* repmat(norm_B(:,2),[1,3]);
    X = [u v norm_A];
 
    % Decompose X
    [U,S,V] = svd(X);
    V_comp = V(:,size(V,2));
    raw_FMtx = reshape(V_comp,[3,3])';
    
    % Enforce S constraints
    [U,S,V] = svd(raw_FMtx);
    S(3,3) = 0;
    S(1,1) = mean([S(1,1), S(2,2)]);
    S(2,2) = mean([S(1,1), S(2,2)]);
    F = U*S*V';
 
    norm_FMtx = K_B'*F*K_A;
     
    end
 
end