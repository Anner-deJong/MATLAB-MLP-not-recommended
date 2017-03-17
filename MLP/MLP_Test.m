function [Loss, ccr] = MLP_Test(X, y, W, BN_bet, BN_gam, lambda, act)

%   Anner de Jong 04/08/2016
%
%   Run after MLP_Train
%   MLP_Test calculates the MLP loss for every W (1 for each iter)

%% Initialise hyperparameters

if ~exist('epsilon', 'var') || isempty(epsilon)
    epsilon = 500;
end

%% Activation functions

% forward none function
none   = @(z) z;

% forward sigmoid function
sig    = @(z) 1 ./ (1 + exp(-z));

% forward ReLU function
ReLU   = @(z) max(z,0);

% used activation functions
f_act = eval( act );

%% Batch Normalization

function [mu, sigma, z_bar] = BN(z)                               %[Ioffe and Szegedy, 2015]
    mu    = mean(z,1);
    sigma = mean(  bsxfun(@minus, z, mu) .^ 2,  1);
    z_bar = bsxfun(@rdivide,  bsxfun(@minus, z, mu),  sigma );
end

%% Initialisation parameters

global type
% model parameters
no_matr       = length(W);
% act_fun     = model.act_fun;

% data parameters
no_samp       = size(X,1);           % number of samples
a             = [ones(no_samp,1),X]; % include bias for first a
[~, corr_ind] = max(y,[],2);         % indices (classes) for actual y_Test
reg           = 0;                   % parameter to calculate regularization

%% Forward pass
    
for j = 1:no_matr

    % sum over squares of the weight matrix for regularization cost
    reg    = reg + sum(sum(W{j}.^2));

    % calculate z
    z    = a * W{j};

    % calculate z_bar (batch normalization)
    [~,~,z_bar] = BN(z);
    z_nor  = bsxfun(@plus,    bsxfun(@times, z_bar, BN_gam{j+1}),    BN_bet{j+1});
    
    % calculate a (activation), add bias
    bias   = ones(size(z_nor,1),1);
    a    = [bias, f_act(z)];

end

scores = z;

%% softmax loss:
    % calculate the exponential fractions
exp_sco = exp(scores);
exp_sum = sum(exp_sco,2);
exp_fra = bsxfun(@rdivide, exp_sco, exp_sum);
    % retrieve the calculated value for the index of the correct score
idL = sub2ind(size(exp_fra), 1:no_samp, corr_ind');  
los_fra = exp_fra(idL)';
    % calculate loss
Loss     = sum(-log(los_fra));
    % correct loss for average
Loss     = Loss / no_samp;
    % correct loss for regularization
reg_cost = 0.5 * lambda * reg;
Loss     = Loss + reg_cost;

%% calculate correct classification rate

[~, calc_ind] = max(z,[],2);                         % indices (classes) for calculated y
diff          = calc_ind - corr_ind;                 % correct classified if this diff = 0
ccr           = sum(diff==0) / no_samp;              % correct classification rate

end


