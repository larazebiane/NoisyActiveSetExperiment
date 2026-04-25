function X_constraints = constraintData(X_train,n_constraints)

% Description : Sets constraint data for trainNeuralNetwork
% Authors     : Lara Zebiane and Frank E. Curtis
% Inputs      : X_train, complete set of training inputs
%               n_constraints, number of desired constraints
% Outputs     : X_constraints, constraint inputs
 
% Randomize indices
[~,ind_constraints] = sortrows(X_train',5);
ind_constraints = ind_constraints';

% Set indices per constraint (except maybe last one)
indices_per_constraint = ceil(size(X_train,2) / n_constraints);

% Set indices for each constraint
indices = cell(n_constraints,1);
for i = 1:n_constraints-1
  indices{i} = ind_constraints((i-1)*indices_per_constraint+1:i*indices_per_constraint);
end
indices{n_constraints} = ind_constraints((n_constraints-1)*indices_per_constraint+1:end);

% Set constraint data
X_constraints = cell(n_constraints,1);
for i = 1:n_constraints
  X_constraints{i} = X_train(:,indices{i});
end
