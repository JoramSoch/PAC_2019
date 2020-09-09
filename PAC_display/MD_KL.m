function KL = MD_KL(p, q)
% _
% Kullback-Leibler divergence
% FORMAT KL = MD_KL(p, q)
% 
%     p  - a 1 x m vector of probabilities (true distribution P)
%     q  - a 1 x m vector of probabilities (estimated distribution Q)
% 
%     KL - a scalar, the Kullback-Leibler divergence of P from Q
% 
% FORMAT KL = MD_KL(p, q) computes the empirical Kullback-Leibler
% divergence between discrete distributions p and q.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 19/08/2020, 15:21


% estimate KL
kl = p .* log(p./q);
kl(isnan(kl)) = 0;
kl(isinf(kl)) = 0;
KL = sum(kl);
clear kl