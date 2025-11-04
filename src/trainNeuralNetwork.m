function trainNeuralNetwork(rho,batch_size,trial)

% Description : Trains neural network for binary classification
% Authors     : Lara Zebiane and Frank E. Curtis
% Inputs      : heart.csv, must exist in current directory
%               rho, penalty parameter
%               batch_size, batch size
%               trial, trial number
% Outputs     : training.out, shows per-iteration output from training

% Set algorithm parameters
split         = 0.8;   % fraction for splitting training and testing data
n_con_max     = 10;    % maximum number of constraints
alpha_1       = 1e-01; % initial step size for line search
alpha_min     = 1e-10; % minimum step size
eta           = 1e-08; % line search sufficient decrease parameter
lbfgs_history = 25;    % LBFGS history
lbfgs_tol     = 1e-06; % LBFGS tolerance for updating
M             = 1e+08; % LP active set parameter
beta          = 1e-08; % LP active set parameter
sigma         = 7e-01; % LP active set parameter
theta         = 5e+00; % QP active set parameter
nu            = 1e+02; % QP active set parameter
QP_tol        = 1e-08; % QP active set parameter
k_max         = 1e+04; % iteration limit for optimization
scale         = 1e-02; % scaling for initial point
tol           = 1e-04; % tolerance for terminating optimization
n_hidden      = 1;     % number of hidden layers
hiddens       = 6;     % nodes in each hidden layer

% Load data
T = readtable('heart.csv','PreserveVariableNames',true);
A = table2array(T);

% Transpose to have features in rows, samples in columns
X = A(:,1:13)';
y = A(:,14)';

% Split indices into training and testing
N         = size(X,2);
[~,ind]   = sortrows(X',5);
ind       = ind';
N_train   = round(split * N);
ind_train = ind(1:N_train);
ind_test  = ind(N_train+1:end);

% Set data points for objective, training and testing
X_train = X(:,ind_train);
y_train = y(:,ind_train);
X_test  = X(:,ind_test);
y_test  = y(:,ind_test);

% Set data points for constraints
n_constraints = min(n_con_max,size(X_train,2));
X_constraints = constraintData(X_train,n_constraints);

% Declare problem
P = constrainedClassification(X_train, y_train, [], X_constraints, n_hidden, hiddens, batch_size);

% Set random seed (for starting point)
rng(19970830);

% Set starting point
w = scale*randn(P.numberOfVariables,1);
P.updateNetworkVariables(w);

% Set random seed (for noisy active set)
rng(batch_size+trial);

% Initialize iteration counter
k = 0;

% Start timing
tic;

% Open output files
output_file_name = "output/training_" + string(rho) + "_" + string(batch_size) + "_" + string(trial) + ".out";
output_file      = fopen(output_file_name,'w');
output_file_name = "output/active_set_LP_" + string(rho) + "_" + string(batch_size) + "_" + string(trial) + ".out";
output_file_LP   = fopen(output_file_name,'w');
output_file_name = "output/active_set_QP_" + string(rho) + "_" + string(batch_size) + "_" + string(trial) + ".out";
output_file_QP   = fopen(output_file_name,'w');

% Print header
fprintf(output_file,'================= TRAINING CONFIGURATION =================\n');
fprintf(output_file,'Number of features            : %d\n', size(X_train,1));
fprintf(output_file,'Training / testing split      : %.1e\n', split);
fprintf(output_file,'Number of training samples    : %d\n', size(X_train,2));
fprintf(output_file,'Number of testing samples     : %d\n', size(X_test,2));
fprintf(output_file,'Number of constraint samples  : %d\n', size(X_constraints,2));
fprintf(output_file,'Initialize step size          : %.1e\n', alpha_1);
fprintf(output_file,'Minimum step size             : %.1e\n', alpha_min);
fprintf(output_file,'Sufficient decrease parameter : %.1e\n', eta);
fprintf(output_file,'LBFGS history                 : %d\n', lbfgs_history);
fprintf(output_file,'LBFGS tolerance               : %d\n', lbfgs_tol);
fprintf(output_file,'Max iterations (k_max)        : %d\n', k_max);
fprintf(output_file,'Initial scale                 : %.1e\n', scale);
fprintf(output_file,'Tolerance (tol)               : %.1e\n', tol);
fprintf(output_file,'Penalty parameter (rho)       : %.1e\n', rho);
fprintf(output_file,'Number of hidden layers       : %d\n', n_hidden);
fprintf(output_file,'Nodes in hidden layers        :');
for i = 1:length(hiddens)
  fprintf(output_file,' %d',hiddens(i));
end
fprintf(output_file,'\n');
fprintf(output_file,'Number of variables           : %d\n', P.numberOfVariables);
fprintf(output_file,'Batch size                    : %d\n', P.batchSize);
fprintf(output_file,'==========================================================\n\n');

% Also print to MATLAB console
fprintf('================= TRAINING CONFIGURATION =================\n');
fprintf('Number of features            : %d\n', size(X_train,1));
fprintf('Training / testing split      : %.1e\n', split);
fprintf('Number of training samples    : %d\n', size(X_train,2));
fprintf('Number of testing samples     : %d\n', size(X_test,2));
fprintf('Number of constraint samples  : %d\n', size(X_constraints,2));
fprintf('Initialize step size          : %.1e\n', alpha_1);
fprintf('Minimum step size             : %.1e\n', alpha_min);
fprintf('Sufficient decrease parameter : %.1e\n', eta);
fprintf('LBFGS history                 : %d\n', lbfgs_history);
fprintf('LBFGS tolerance               : %d\n', lbfgs_tol);
fprintf('Max iterations (k_max)        : %d\n', k_max);
fprintf('Initial scale                 : %.1e\n', scale);
fprintf('Tolerance (tol)               : %.1e\n', tol);
fprintf('Penalty parameter (rho)       : %.1e\n', rho);
fprintf('Number of hidden layers       : %d\n', n_hidden);
fprintf('Nodes in hidden layers        :');
for i = 1:length(hiddens)
  fprintf(' %d',hiddens(i));
end
fprintf('\n');
fprintf('Number of variables           : %d\n', P.numberOfVariables);
fprintf('Batch size                    : %d\n', P.batchSize);
fprintf('==========================================================\n\n');

% Initialize LBFGS pairs
sss = [];
yyy = [];

% Main loop
while true

  % Print information line
  if mod(k,20) == 0
    fprintf(output_file,'%6s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s\n',...
      'k','f','f_total','|cE|','|max{cI,0}|','|d|','Accuracy','||g_f||','||g_total||','alpha','min(cI)','LP_thresh','min(cJd_QP)','max(cJd_QP)');
    fprintf('%6s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s\n',...
      'k','f','f_total','|cE|','|max{cI,0}|','|d|','Accuracy','||g_f||','||g_total||','alpha','min(cI)','LP_thresh','min(cJd_QP)','max(cJd_QP)');
  end

  % Compute function values
  [f,g_f] = P.objectiveFunctionAndGradient(w,0);
  [cE,JE] = P.constraintFunctionAndJacobianEqualities(w,0);
  [cI,JI] = P.constraintFunctionAndJacobianInequalities(w,0);

  % Check trial number
  if trial == 0

    % Compute active-set estimates
    [active_set_LP,threshold_LP] = estimateActiveSetLP(g_f,cE,JE,cI,JI,M,beta,sigma);
    [active_set_QP,cJd_QP]       = estimateActiveSetQP(g_f,cE,JE,cI,JI,theta,nu,QP_tol);

  else

    % Compute noisy function values
    [f_noisy,g_f_noisy] = P.objectiveFunctionAndGradient(w,1);
    [cE_noisy,JE_noisy] = P.constraintFunctionAndJacobianEqualities(w,1);
    [cI_noisy,JI_noisy] = P.constraintFunctionAndJacobianInequalities(w,1);

    % Compute active-set estimates
    [active_set_LP,threshold_LP] = estimateActiveSetLP(g_f_noisy,cE_noisy,JE_noisy,cI_noisy,JI_noisy,M,beta,sigma);
    [active_set_QP,cJd_QP]       = estimateActiveSetQP(g_f_noisy,cE_noisy,JE_noisy,cI_noisy,JI_noisy,theta,nu,QP_tol);

  end

  % Print active set estimates
  for i = 1:length(active_set_LP)
    fprintf(output_file_LP,' %d',active_set_LP(i));
  end
  fprintf(output_file_LP,'\n');
  for i = 1:length(active_set_QP)
    fprintf(output_file_QP,' %d',active_set_QP(i));
  end
  fprintf(output_file_QP,'\n');

  % Compute penalty terms
  penalty_term     = 0.5 * rho * norm(max(cI, 0), 2)^2;
  penalty_gradient = rho * (JI' * max(cI, 0));

  % Compute penalty function
  f_total = f + penalty_term;
  g_total = g_f + penalty_gradient;

  % Compute predictions
  y_pred  = P.predictions(X_train,w);
  y_pred  = (y_pred >= 0.5);
  correct = length(find(y_train == y_pred));

  % Check for termination
  if k >= k_max || norm(g_total,inf) <= tol
    break;
  end

  % Compute search direction
  if size(sss,2) == 0

    % Steepest descent if no history
    d = -g_total;

  else

    % LBFGS if history
    q = g_total;
    for i = size(sss,2):-1:1
      lambda(i) = 1/(sss(:,i)'*yyy(:,i));
      zeta(i) = lambda(i)*sss(:,i)'*q;
      q = q - zeta(i)*yyy(:,i);
    end
    z = q;
    for i = 1:size(sss,2)
      bbbeta(i) = lambda(i)*yyy(:,i)'*z;
      z = z + sss(:,i)*(zeta(i) - bbbeta(i));
    end
    d = -z;

  end

  % Run line search
  alpha = alpha_1;
  while true

    % Set trial point
    w_trial = w + alpha * d;
    P.updateNetworkVariables(w_trial);

    % Compute function values
    [f_trial,g_f_trial] = P.objectiveFunctionAndGradient(w_trial,0);
    [cE_trial,JE_trial] = P.constraintFunctionAndJacobianEqualities(w_trial,0);
    [cI_trial,JI_trial] = P.constraintFunctionAndJacobianInequalities(w_trial,0);

    % Compute penalty terms
    penalty_term_trial     = 0.5 * rho * norm(max(cI_trial, 0), 2)^2;
    penalty_gradient_trial = rho * (JI_trial' * max(cI_trial, 0));

    % Compute penalty function
    f_total_trial = f_trial + penalty_term_trial;
    g_total_trial = g_f_trial + penalty_gradient_trial;

    % Check for decrease
    if alpha < alpha_min || f_total_trial <= f_total + alpha * eta * g_total' * d
      break
    else
      alpha = alpha / 2;
    end
  end

  % Update LBFGS pairs
  if (w_trial - w)'*(g_total_trial - g_total) >= lbfgs_tol * norm(w_trial - w) * norm(g_total_trial - g_total)
    sss = [sss (w_trial - w)];
    yyy = [yyy (g_total_trial - g_total)];
  end
  if size(sss,2) > lbfgs_history
    sss = sss(:,2:end);
    yyy = yyy(:,2:end);
  end

  % Update iterate
  w = w_trial;

  % Write output line to file
  fprintf(output_file,'%6d %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e\n',...
    k, f, f_total, norm(cE,inf), norm(max(cI,0),inf), norm(d,inf), correct/length(y_train), norm(g_f, inf), norm(g_total, inf), alpha, min(cI), threshold_LP, min(cJd_QP), max(cJd_QP));

  fprintf('%6d %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e\n',...
    k, f, f_total, norm(cE,inf), norm(max(cI,0),inf), norm(d,inf), correct/length(y_train), norm(g_f, inf), norm(g_total, inf), alpha, min(cI), threshold_LP, min(cJd_QP), max(cJd_QP));

  % Saving 'w'
  save("solutions/w_" + string(rho) + ".mat","w");

  % Increment iteration counter
  k = k + 1;

end

% Compute predictions on test data
y_test_pred = P.predictions(X_test, w);
y_test_pred = (y_test_pred >= 0.5);
correct_test = sum(y_test == y_test_pred);
accuracy_test = correct_test / length(y_test);

% Print final test accuracy
fprintf('\nFinal TEST accuracy : %+.4e\n', accuracy_test);
fprintf(output_file, '\nFinal TEST accuracy : %+.4e\n', accuracy_test);

% Print final accuracy to Matlab command window, to see final result
fprintf('\nFinal accuracy achieved on training data: %+.4e\n', correct / length(y_train));
fprintf(output_file, '\nFinal accuracy achieved on training data: %+.4e\n', correct / length(y_train));

% Print total time
elapsed_time = toc;
fprintf('\nTotal training time: %.2f seconds\n', elapsed_time);
fprintf(output_file, '\nTotal training time: %.2f seconds\n', elapsed_time);

% Close output files
fclose(output_file);
fclose(output_file_LP);
fclose(output_file_QP);