function runExperiment

% Description : Runs experiments for paper
% Authors     : Lara Zebiane and Frank E. Curtis

% Set number of trials
num_trials = 10;

% Set batch sizes
batch_size = 32;

% Set penalty parameter for unconstrained
rho = 0;

% Train neural network
trainNeuralNetwork(rho,0,0);

% Set penalty parameter for constrained
rho = 1e+02;

% Train neural network
trainNeuralNetwork(rho,0,0);

% Generate plots
plotActiveSetResults(0,'full-batch',1);

% Loop over batch sizes
for b = 1:length(batch_size)

  % Loop over trials
  for trial = 1:num_trials

    % Train neural network
    trainNeuralNetwork(rho,batch_size(b),trial);

  end

  % Generate plots
  plotActiveSetResults(1,batch_size(b),num_trials);

end