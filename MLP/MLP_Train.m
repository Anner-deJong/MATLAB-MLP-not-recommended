function [model] = MLP_Train(X, Y, X_Test, Y_Test, iter, batch_size, lambda, lrn_rate, decay, no_layers, act)

%   Anner de Jong 06/07/2016
%
%   MLP_Trains trains an MLP with given regularization and learning rate
%   Requires a seperate Loss and Gradient calculator called MLP_LG

%% Initialisation hyperparameters

if ~exist('no_layers', 'var') || isempty(no_layers) || no_layers < 1        % Number of layers
    no_layers = 2;
end

if ~exist('iter', 'var') || isempty(iter)                                   % Number of iterations
    iter = 300;
end

if ~exist('lambda', 'var') || isempty(lambda)                               % Regularization strength
    lambda = 0.1;
end

if ~exist('lrn_rate', 'var') || isempty(lrn_rate)                           % Learning rate
    lrn_rate = exp(-7);
end

if ~exist('batch_size', 'var') || isempty(batch_size) || ...                % batch size
        (batch_size >= size(X,1));
    batch_size = size(X,1);
end

%% Initialisation parameters

global type                                                 % type of prediction, regression or classification

X_space  = size(X,2);                                       % size of input parameters
y_space  = size(Y,2);                                       % number of classes

no_samp  = size(X,1);                                       % number of samples

log_X  = log10(X_space);                                     % log size of input parameters
log_y  = log10(y_space);                                     % log number of classes
incr   = (log_y-log_X)/(no_layers + 1);                      % log size increment per layer

Loss   = zeros(1,iter);                                       % initialize empty loss history
CCR    = zeros(1,iter);                                       % initialize empty ccr history

Loss_Test = zeros(1,iter);                                  % initialize empty loss history
CCR_Test  = zeros(1,iter);                                  % initialize empty ccr history

%% Calculating hidden layer sizes

lay_space = zeros(1,no_layers);                             % no of units per hidden layer

for i = 1:(no_layers)
    lay_space(i) = round(10^(log_X+incr * i));
end

%% Initialisation weight matrix + Batch Normalization parameters


NB_bet = cell(1,no_layers + 2);          % Batch Normalization parameter
NB_gam = cell(1,no_layers + 2);          % Batch Normalization parameter

W      = cell(1,no_layers + 1);          % initialize empty weight matrices cell
rng('default');                          % control seed for randomness (Mersenne Twister with seed 0)

for i = 1:(no_layers + 1)
    
    if i == 1     % FIRST LAYER
        W{i}      = normrnd(0,1,  [X_space + 1, lay_space(i)]  );  % randomly assign weights
        W{i}      = W{i}      .*    sqrt( 2 / (X_space + 1)    );  % Initisialisation accord. to [He et Al, 2015]            % resize the weights
        
    elseif i == (no_layers + 1)    % LAST LAYER
        W{i}      = normrnd(0,1,       [lay_space(i-1) + 1, y_space]  );             
        W{i}      = W{i}      .*   sqrt( 2 / (lay_space(i-1) + 1)     );
        
    else          % MIDDLE LAYER(S)
        W{i}      = normrnd(0,1,  [lay_space(i-1) + 1, lay_space(i)]  );
        W{i}      = W{i}      .*   sqrt( 2 / (lay_space(i-1) + 1)     );
    end
   
    NB_bet{i+1}   = zeros( 1, size(W{i},2));        % Initialise betas as zeros
    NB_gam{i+1}   = ones ( 1, size(W{i},2));        % Initialise gammas as ones
    
end

%% Iteration

for i = 1:iter
    
    % calculate loss, gradients and ccr for test data   
    [L_Test, ccr_Test] = MLP_Test(X_Test, Y_Test, ...
                                   W, NB_bet, NB_bet, lambda, act);
    Loss_Test(i)       = L_Test;                               % update Test loss history
    CCR_Test(i)        = ccr_Test;                             % update Test ccr history       
    
    % create batch to speed up calculation
    Tr_samples = 1:no_samp;
    batch_Ind = randsample(Tr_samples,batch_size);   
    X_batch = X(batch_Ind', :);
    Y_batch = Y(batch_Ind', :);

    % calculate loss and gradient for training data
    [loss, grad, ccr] = MLP_LG(X_batch, Y_batch, ...   % calculate loss, gradients and ccr
                                W, NB_bet, NB_gam, lambda, act);
    
    Loss(i) = loss;                                            % update loss history
    CCR(i)  = ccr;                                             % update ccr history
    
    for j = 1:(no_layers+1)
        W{j}      = W{j}      - (lrn_rate * grad.W{j});        % update weight matrix
        NB_bet{j} = NB_bet{j} - (lrn_rate * 0 * grad.bet{j});      % update batch normalization: beta  parameter
        NB_gam{j} = NB_gam{j} - (lrn_rate * 0 * grad.gam{j});      % update batch normalization: gamma parameter
    end
    
    % update learning rate with decay
    epoch    = floor(i/(no_samp/batch_size));
    lrn_rate = lrn_rate * decay^epoch;
    
end

%% output

% initialize empty model structure
model                = struct;

% loss, weights and ccr for testing
model.loss           = Loss(iter);
model.weights        = W;
model.ccr            = CCR(iter);

% general parameters
model.regul_strength = lambda;
model.learning_rate  = lrn_rate;
model.hidd_lay_no    = no_layers;
model.hidd_lay_size  = lay_space;  % returns array of length number of hidden layers, with each value representing the size of that layer. 

% activation and Batch Normalization
model.BN.gamma       = NB_gam;
model.BN.beta        = NB_bet;
model.act_fun        = act;


% history
model.loss_hist      = Loss;
model.ccr_hist       = CCR;

% Test data
model.loss_Test_hist = Loss_Test;
model.ccr_Test_hist  = CCR_Test;

%% output documentation

% model.hidd_lay_size
% returns array of length number of hidden layers, with each value
% representing the size of that layer. 

end

