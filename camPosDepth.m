% https://www.mathworks.com/help/vision/ref/relativecamerapose.html?s_tid=doc_ta

function [Rot, Tsn] = camPosDepth(E_Rot, E_Tsn, K, inlierPts1, inlierPts2)
    data = zeros(1,4);
    
    
    for i = 1:size(E_Rot, 3)
        % 4x3 RT'*K matrix --> image to world
        RTK_2 = [E_Rot(:,:,i)';E_Tsn(i,:)]*K;
        tx = triangulate3D(K, RTK_2, inlierPts1, inlierPts2);
        ty = tx*E_Rot(:,:,i)' + E_Tsn(i, :);
        
        data(i) = sum((tx(:,3)<0 | (ty(:,3)<0)));
        
    end
    % Find index of minimum value; 3/4 will be greater --> non-solutions
    [Y, index] = min(data);
    
    % Return Rotation and Tranlation with positive depth (min value index)
    Rot = E_Rot(:,:,index);
    Tsn = E_Tsn(index, :)*Rot;
    
    function points3D = triangulate3D(intrinsic, RTK_B, points1, points2)
        % https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf
        % https://avisingh599.github.io/vision/visual-odometry-full/
        
        numPts = size(points1, 1);
        points3D = zeros(numPts, 3);
        
        %% Image A
        % Append 1 to inlier points 
        pts1Padded = horzcat(points1, ones(numPts ,1))';
        RTK_A = [eye(3);[0 0 0]]*intrinsic;
        RTK_A = RTK_A';
        mid_A = RTK_A(1:3,1:3);
        % Ratio of R in RTK to T
        A = mldivide(-mid_A,RTK_A(:,4));
        azc_A = mldivide(mid_A,pts1Padded); 
        
        %% Image B
        pts2Padded = horzcat(points2, ones(numPts ,1))';
        RTK_B = RTK_B';
        mid_B = RTK_B(1:3, 1:3);
        B = mldivide(-mid_B,RTK_B(:,4));
        azc_B = mldivide(mid_B, pts2Padded);
        
        diff = B - A;
        
        % Sturm algorithm iterates over number inlier points
        for jj = 1:numPts
            F = [azc_A(:,jj),-azc_B(:,jj)];  
            D = mldivide(F,diff);        
            p = (A + D(1)*azc_A(:,jj) + B + D(2) * azc_B(:,jj)) / 2;
            points3D(jj, :) = p';
        end
    end
end

