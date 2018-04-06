function [Rot, Tsn] = EMtx2RT(E, W, Z)
    % Adhere S matrix to constraints
    [U, S, V] = svd(E);
    S(3,3) = 0;
    % Ensure these two values are non-unique
    S(1,1) = mean([S(1,1), S(2,2)]);
    S(2,2) = mean([S(1,1), S(2,2)]);
    
    % Reform Essential Matrix, fully constrained
    E = U*S*V';
    
    % Translation vector
    t = U(:,3)';
    
    % Rotation matrices
    Rot1 = U*W*V';
    Rot2 = U*W'*V';
    
    % CONSTRAINT: det(Rot) = 1
    if det(Rot1) < 0
        Rot1 = -Rot1;
    end
    if det(Rot2) < 0
        Rot2 = -Rot2;
    end

    % Output four possible pairs for Rotation|Translation
    Rot = cat(3, Rot1, Rot1, Rot2, Rot2);
    Tsn = cat(1, t,    -t,   t,    -t);
end