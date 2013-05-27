function [U, V, J_history] = GradientDescent(R,S,theta,alpha,lambda_U,lambda_V,K,l,num_iters)
% CopyRight @ Jinfeng Rao 2013/04/24
% Input:
% R : rating matrix, with m users and n movies 
% S : user similarity matrix, m*m dimensional
% thata : learning rate
% alpha : weight of score from rating matrix R 
% lambda_u : regularized parameter for user vector
% lambda_v : regularized parameter for user vector
% K : top-K friend for each user
% l : dimension of user and item latent factor
% num_iters: 
% Output:
% U : user latent matrix, m*l dimensional
% V : item latent matrix, n*l dimensional
% J_history : cost function value in each iteration

% 把参数和实验结果写入文件
fid = fopen('D:\ThesisData\code\alpha2.txt','a+');
fprintf(fid,'%s%6.4f %s%d %s%d\r\n','theta = ',theta,'K = ',K,'l = ',l);
fprintf(fid,'%s %2.1f\r\n','alpha = ',alpha);

J_history = zeros(1,num_iters+1);
R = double(R)/5.0;
[m,n] = size(R);
U = 0.5 * rand(m,l);
V = 0.5 * rand(n,l);
old_U = U;
old_V = V;
MAE_Val = CompMAE(R,S,U,V,K,alpha);
RMSE_Val = CompRMSE(R,S,U,V,K,alpha)
fprintf(fid,'%s%6.4f %s%6.4f\r\n','MAE = ',MAE_Val,'RMSE = ',RMSE_Val);
for iter = 1:num_iters
    iter
    tic
    for i = 1:m
       ratedIndex = R(i,:)~=0;
       friends_rating = zeros(1,n); % variable part1 and part2 are both intermediate value in the computing process
       
        for k = 1:K
            index = S(i,k).index(1);
            sim = S(i,k).value;
            friends_rating = friends_rating + sim*U(index,:)*V'; % compute score using similarity matrix S
        end
        pmf_rating = U(i,:) *V';
        sumVec = alpha * ratedIndex .* (alpha * pmf_rating + (1-alpha)*friends_rating - R(i,:)) ;
        product = sumVec * V;
        derivative = product + lambda_U * U(i,:);       
        old_U(i,:) = U(i,:) - theta * derivative;
    end
    toc
    
    S2 = struct2cell(S);
    value_matrix = cell2mat(reshape(S2(1,:,:),[m K]));
    index_matrix = cell2mat(reshape(S2(2,:,:),[m K]));
    
    tic
    for j = 1:n      
      ratedIndex = R(:,j)~=0;
      friends_rating = zeros(1,m); % variable part1 and part2 are both intermediate value in the computing process
      friends = zeros(m,l);
      
      for k = 1:K
          index = index_matrix(:,k); % get 1*2000 vector
          sim = value_matrix(:,k);
          sim2 = repmat(sim,1,l); %concatence vector sim' with l times, get m*l matrix
          friends_rating = friends_rating + sim' .* (V(j,:) * U(index,:)'); % compute score using similarity matrix S
          friends = friends + sim2 .* U(index,:);
      end
      pmf_rating = V(j,:) * U';    
      sumVec = ratedIndex' .* (alpha * pmf_rating + (1-alpha)*friends_rating - R(:,j)') ;
      product = sumVec * ( alpha * U + (1-alpha) * friends);
      derivative = product + lambda_V * V(j,:);
      old_V(j,:) = V(j,:) - theta * derivative;      
    end
    toc    
   
    U = old_U;
    V = old_V;    
    MAE_Val = CompMAE(R,S,U,V,K,alpha);
    RMSE_Val = CompRMSE(R,S,U,V,K,alpha)
    fprintf(fid,'%s%d %s%6.4f %s%6.4f\r\n','iter = ',iter, ': MAE = ',MAE_Val,'RMSE = ',RMSE_Val);

end
fprintf(fid,'\n');
fprintf(fid,'\n');
fclose(fid);