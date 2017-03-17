function [Loss, grad, ccr] = MLP_LG(X, Y, W, BN_bet, BN_gam, lambda, act)

%   Anner de Jong 06/07/2016
%
%   MLP_LG calculates the Loss and the Gradients for an MLP machine
%   learning algorithm, using backpropagation and the following cost/Loss
%   function:
%   Softmax loss:
%   J = 1/m * sum_over_m *
%           -log (   y .* exp(h_k)  /  (  sum_over_K (exp(h_k))  )   )
%   y .* exp(h_k) = basically array of calculated values for the correct classes
%   K being the index for no of classes

%   or (currently not implemented):
%   J = 1/m * sum_over_m *
%           sum_over_K (-y_k * log(h_k) - (1-y_k) * log(1-h_k))
%   h being the final computed output layer
%   =
%   J = 1/no_samp * -sum(sum(y .* log(y_calc) + (1-y) .* log(1-y_calc)));

%% Initialisation parameters

global type
no_matr      = length(W);
no_samp      = size(X,1);
[~,corr_ind] = max(Y,[],2);            % identify the correct score indices

reg          = 0;                      % variable to calculate the regularization loss
z            = cell(1, size(W,2) + 1); % |
z_nor        = cell(1, size(W,2) + 1); % |hidden layers' values:  z = before activation, z_bar = normalized z, a = activated
a            = cell(1, size(W,2) + 1); % |

dW           = cell(size(W));          % empty cell for W        gradients
dBN_bet      = cell(size(BN_bet));     % empty cell for BN beta  gradients
dBN_gam      = cell(size(BN_gam));     % empty cell for BN gamma gradients

%% Activation functions

% forward none function
none   = @(z) z;

% backward none function
d_none = @(z, da) da;

% forward sigmoid function
sig    = @(z) 1 ./ (1 + exp(-z));

% backward sigmoid function
d_sig  = @(z, da) (sig(z) .* (1 - sig(z))) .* da;

% forward ReLU function
ReLU   = @(z) max(z,0);

% backward ReLU function
    function dz = d_ReLU_help(z, da) 
        da(z<=0) = 0;
        dz       = da;
    end
d_ReLU = @(z, da) d_ReLU_help(z, da);

% used activation functions
f_act = eval(             act  );
d_act = eval( strcat('d_',act) );

%% Batch Normalisation

function [mu, sigma, z_bar] = BN(z)                               %[Ioffe and Szegedy, 2015]
    mu    = mean(z,1);
    sigma = mean(  bsxfun(@minus, z, mu) .^ 2,  1);
    z_bar = bsxfun(@rdivide,  bsxfun(@minus, z, mu),  sigma );
end

function [dz, d_bet, d_gam] = d_BN(z, dz_nor, gamma) %[Ioffe and Szegedy, 2015]
    % help parameters
    [mu, sigma, zbar] = BN(z);
    m          = size(dz_nor,1);
    spread     = bsxfun(@minus, z, mu);
    dz_bar     = bsxfun(@times, dz_nor, gamma);

    % beta & gamma
    d_bet = sum( dz_nor         , 1);   % beta  gradients 
    d_gam = sum( dz_nor .* zbar , 1);   % gamma gradients    

    % sigma
    d_sig_1    = dz_bar .* spread;
    d_sig_2    = - 0.5 * sigma .^(-3/2);
    d_sigm     = sum(  bsxfun(@times, d_sig_1, d_sig_2)  , 1);
    
    % mu
    d_mu_1     = sum(  bsxfun(@times, dz_bar, -1 ./ sqrt( sigma )  )  , 1);
    d_mu_2     =       bsxfun(@times, -2 * mean(spread,1) ,  d_sigm );
    d_mu       = d_mu_1 + d_mu_2;
    


    % pre normalized layer
    dz_1       = bsxfun(@times, dz_bar, 1 ./ sqrt( sigma ) );
    dz_2       = bsxfun(@times, 2/m * spread ,  d_sigm );
    dz_3       = (d_mu / m);

    dz         = bsxfun(@plus, dz_1 + dz_2, dz_3);
end

%% Forward pass

a{1}  = [ones(no_samp,1),X];    % initial a for the subsequent for loop

for i = 1:no_matr
    
    % sum over squares of the weight matrix for regularization cost
    reg    = reg + sum(sum(W{i}.^2));
    
    % calculate z
    z{i+1} = a{i} * W{i};
    
    % calculate z_bar (batch normalization)
    [~,~,z_bar] = BN(z{i+1});
    z_nor{i+1}  = bsxfun(@plus,    bsxfun(@times, z_bar, BN_gam{i+1}),    BN_bet{i+1});
    
    % calculate a (activation), add bias
    bias   = ones(size(z_nor{i+1},1),1);
    a{i+1} = [bias, f_act(z{i+1})];
    
end

scores = z{no_matr + 1};

%% Softmax loss

% calculate the exponential fractions
exp_sco = exp(scores);

exp_sum = sum(exp_sco,2);
exp_fra = bsxfun(@rdivide, exp_sco, exp_sum);

% 10 layers goes to inf, because of -log(exp_fra)
%sum(sum(isinf(-log(exp_fra))))

idL = sub2ind(size(exp_fra), 1:no_samp, corr_ind');  % retrieve the calculated value
los_fra = exp_fra(idL)';                             % for the index of the correct score


% calculate loss
Loss     = sum(-log(los_fra));

    % correct for average
Loss     = Loss / no_samp;

    % correct for regularization
reg_cost = 0.5 * lambda * reg;
Loss     = Loss + reg_cost;

%% Correct classification rate

[~, calc_ind] = max(scores,[],2);                    % indices (classes) for calculated y
diff          = calc_ind - corr_ind;                 % correctly classified if this diff = 0
ccr           = sum(diff==0) / no_samp;              % correct classification rate

%% Backward pass, softmax part

% dL_dfra: calculate the derivative of L to the fraction of correct y indices
dL_dfra = (-1 ./ los_fra);

% dL_dsco: calculate the derivative of the fraction to calculated 'scores' matrix, times dL_dfra
dL_dsco = bsxfun(@times, exp_fra,  (-los_fra .* dL_dfra)  );

% dL_dsco: correction for the scores with correct indices 
dL_dsco(idL) = (los_fra - los_fra .^ 2) .* dL_dfra;

%% Backward pass, through hidden layers, calculating gradients on the way

% dW: calculate the derivative of the calculated 'scores' matrix to the weight matrix W, times dl_dsco

dz = dL_dsco;               % initial dz

for i = 1:(no_matr)
    
    % reverse the index
    j     = no_matr-(i-1);
    
    % calculate weight gradient
    dW{j} = a{j}' * dz;
    
    % remove bias
    da    = dz * W{j}';
    da    = da(:,2:end);
    
    % calculate z & batch normalization gradients
    if j~=1
        
        dz_nor    = d_act(z{j}, da);
        [dz, d_bet, d_gam] = d_BN(z{j}, dz_nor, BN_gam{j});
        
        dBN_bet{j} = d_bet;   % BN beta  gradients [Ioffe and Szegedy, 2015]
        dBN_gam{j} = d_gam;   % BN gamma gradients [Ioffe and Szegedy, 2015]
%         
%         mu         = mean(z{j},1);
%         spread     = bsxfun(@minus, z{j}, mu);
%         dz_bar     = bsxfun(@times, dz_nor, BN_gam{j});
%         sigma      = mean( spread .^ 2,  1);
%         
%         d_sig_1    = dz_bar .* spread;
%         d_sig_2    = - 0.5 * sigma .^(-3/2);
%         d_sig      = sum(  bsxfun(@times, d_sig_1, d_sig_2)  , 1);
% 
%         d_mu_1     = sum(  bsxfun(@times, dz_bar, -1 ./ sqrt( sigma )  )  , 1);
%         d_mu_2     =       bsxfun(@times, -2 * mean(spread,1) ,  d_sig );
%         d_mu       = d_mu_1 + d_mu_2;
%         
%         m          = size(dz_nor,1);
%         dz_1       = bsxfun(@times, dz_bar, 1 ./ sqrt( sigma ) );
%         dz_2       = bsxfun(@times, 2/m * spread ,  d_sig );
%         dz_3       = (d_mu / m);
%         
%         dz         = bsxfun(@plus, dz_1 + dz_2, dz_3);
        
    end
    
    % correct weight gradient for average and regularization
    dW{j} = dW{j} ./ no_samp;
    dW{j} = dW{j} + lambda .* W{j};
    
end

grad     = struct;
grad.W   = dW;
grad.bet = dBN_bet;
grad.gam = dBN_gam;
end

