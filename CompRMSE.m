function [RMSE] = CompRMSE(R_Testing,S,U,V,K,alpha)

[m,n] = size(R_Testing);
R_predict = zeros(m,n);
% Compute R Predict rating
for i = 1:m
   pmf_rating = U(i,:)*V';
   friends_rating = zeros(1,n);
   for k = 1:K
       sim = S(i,k).value;
       index = S(i,k).index(1);
       friends_rating = friends_rating + sim * U(index,:)*V';
   end
   R_predict(i,:) = alpha * pmf_rating + (1-alpha) * friends_rating;
end

RMSE =0;
ratedNum = 0;
for i =1:m
    ratedMovie = R_Testing(i,:)~=0;
    ratedNum = ratedNum + sum(ratedMovie);
    RMSE = RMSE + sum((5*(R_Testing(i,ratedMovie) - R_predict(i,ratedMovie))).^2);
end
RMSE = sqrt(RMSE/ratedNum);