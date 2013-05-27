function [U,V,RMSE] = PMF(R,theta,lambda_u,lambda_v,l,num_iters)
R = double(R)/5.0;
[m,n] = size(R);
U = 0.5 * rand(m,l);
V = 0.5 * rand(n,l);
old_U = U;
old_V = V;

MAE = CompPMF_MAE(R,U,V)
RMSE(1) = CompPMF_RMSE(R,U,V);
count=2;    
for iter = 1:num_iters
   
    for i = 1:m
       ratedIndex1 = R(i,:)~=0 ;
       sumVec1 = ratedIndex1 .* (U(i,:) * V' - R(i,:));
       product1 = sumVec1 * V;
       derivative1 = product1 + lambda_u * U(i,:);
       old_U(i,:) = U(i,:) - theta * derivative1;
    end
    
    for j = 1:n
       ratedIndex2 = R(:,j)~=0;
       sumVec2 = ratedIndex2 .* (U * V(j,:)' - R(:,j));
       product2 = sumVec2' * U;
       derivative2 = product2 + lambda_v * V(j,:);
       old_V(j,:) = V(j,:) - theta * derivative2;
    end
    
    U = old_U;
    V = old_V;
    MAE = CompPMF_MAE(R,U,V)
    RMSE(count) = CompPMF_RMSE(R,U,V);
    count = count + 1;
end
