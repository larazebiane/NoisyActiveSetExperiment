classdef constrainedClassification < handle

  % Description : Neural network, objective, and constraint function
  %               definitions for trainNeuralNetwork
  % Authors     : Lara Zebiane and Frank E. Curtis

  % Properties
  properties  

    batch_size      % batch size
    inputs_obj      % inputs for objective (the feature data)
    inputs_conE     % inputs for constraints, equalities
    inputs_conI     % inputs for constraints, inequalities
    network         % neural network
    n_features      % number of features
    n_hidden        % number of hidden layers
    n_hidden_nodes  % number of nodes in each hidden layer (vector w/ length n_hidden)
    n_outputs       % number of outputs
    n_variables     % number of variables (learnable network values)
    outputs_obj     % outputs for objective (the true labels)
    w_last          % last primal variable
    w_table         % table of variables

  end % properties

  % Methods
  methods (Access = public)

    % Batch size
    function b = batchSize(self)

      % Batch size
      b = self.batch_size;

    end % batchSize

    % Constructor
    function self = constrainedClassification(ins_obj,outs_obj,ins_conE,ins_conI,n_hidden,n_hidden_nodes,batch_size)

      % Set batch size
      self.batch_size = min([batch_size, size(ins_obj,2)]);

      % Set objective data (inputs and outputs)
      self.inputs_obj  = dlarray(ins_obj,'CB');
      self.outputs_obj = dlarray(outs_obj,'CB');

      % Set constraint data (only inputs)
      self.inputs_conE = dlarray(ins_conE,'CB');
      self.inputs_conI = cell(length(ins_conI),1);
      for i = 1:length(ins_conI)
        self.inputs_conI{i} = dlarray(ins_conI{i},'CB');
        self.batch_size = min(self.batch_size,size(self.inputs_conI{i},2));
      end

      % Set input and output sizes
      self.n_features = size(ins_obj, 1);
      self.n_outputs  = size(outs_obj,1);

      % Set hidden layer sizes
      self.n_hidden = n_hidden;
      for i = 1:self.n_hidden
        self.n_hidden_nodes(i) = n_hidden_nodes(i);  
      end

      % Set number of optimization variables
      if self.n_hidden == 0
        self.n_variables = self.n_features * self.n_outputs + self.n_outputs;
      else
        self.n_variables = self.n_features * self.n_hidden_nodes(1) + self.n_hidden_nodes(1);
        for i = 2:self.n_hidden
          self.n_variables = self.n_variables + self.n_hidden_nodes(i-1) * self.n_hidden_nodes(i) + self.n_hidden_nodes(i);
        end
        self.n_variables = self.n_variables + self.n_hidden_nodes(end) * self.n_outputs + self.n_outputs;
      end
      
      % Set weight and bias terms
      if self.n_hidden == 0
        W{1} = zeros(self.n_outputs,self.n_features);
        b{1} = zeros(self.n_outputs,1);
      else
        W{1} = zeros(self.n_hidden_nodes(1),self.n_features);
        b{1} = zeros(self.n_hidden_nodes(1),1);
        for i = 2:self.n_hidden
          W{i} = zeros(self.n_hidden_nodes(i),self.n_hidden_nodes(i-1));
          b{i} = zeros(self.n_hidden_nodes(i),1);
        end
        W{self.n_hidden+1} = zeros(self.n_outputs,self.n_hidden_nodes(end));
        b{self.n_hidden+1} = zeros(self.n_outputs,1);
      end

      % Initialize neural network
      self.network = initializeNeuralNetwork(self,W,b);

      % Set table value
      for i = 1:self.n_hidden+1
        table_value{2*(i-1)+1,1} = W{i};
        table_value{2*(i-1)+2,1} = b{i};
      end

      % Set variable table
      self.w_table = table(self.network.Learnables.Layer,self.network.Learnables.Parameter,table_value,'VariableNames',{'Layer','Parameter','Value'});

      % Initialize last primal variable
      self.w_last = zeros(self.n_variables,1);

    end % constrainedClassification

    % Constraint function and Jacobian, equalities
    function [cE,JE] = constraintFunctionAndJacobianEqualities(self,w,option)

      % Update network variables
      self.updateNetworkVariables(w);

      % Check type
      if option == 0 || size(self.inputs_conE,2) == 0
        ins = self.inputs_conE;
      else
        indices = round(1 + (size(self.inputs_conE,2)-1)*rand(1,self.batch_size));
        ins = dlarray(self.inputs_conE(:,indices),'CB');
      end

      % Evaluate constraint function and Jacobian, equalities
      [cE,JE] = dlfeval(@self.evaluateConstraintFunctionAndJacobianEqualities,self.network,ins);

    end % constraintFunctionEqualities

    % Constraint function and Jacobian, inequalities
    function [cI,JI] = constraintFunctionAndJacobianInequalities(self,w,option)

      % Update network variables
      self.updateNetworkVariables(w);

      % Check type
      if option == 0 || size(self.inputs_conI,2) == 0
        ins = self.inputs_conI;
      else
        for i = 1:length(self.inputs_conI)
          indices = round(1 + (size(self.inputs_conI,2)-1)*rand(1,self.batch_size));
          ins{i} = dlarray(self.inputs_conI{i}(:,indices),'CB');
        end
      end

      % Evaluate constraint function and Jacobian, inequalities
      [cI,JI] = dlfeval(@self.evaluateConstraintFunctionAndJacobianInequalities,self.network,ins);

    end % constraintFunctionInequalities

    % Evaluate constraint function and Jacobian value, equalities
    function [cE,JE] = evaluateConstraintFunctionAndJacobianEqualities(self,network,inputs)

      % TO DO
      cE = []; JE = [];

    end % evaluateConstraintFunctionAndJacobianEqualities
    
    % Evaluate constraint function and Jacobian value, inequalities
    function [cI, JI] = evaluateConstraintFunctionAndJacobianInequalities(self, network, inputs)

      % Loop over batches of inputs
      for i = 1:length(inputs)

        % Grab input batch
        ins = inputs{i};

        % Set dlarray format
        ins = dlarray(ins,'CB');

        % Select cholesterol feature index
        chol_index = 5;

        % Create function handle that outputs scalar loss (mean y_pred)
        % This allows `dlgradient` to compute vector-Jacobian product
        chol_input = ins(chol_index,:);

        % Enable auto-diff on chol_input (cholesterol only)
        chol_input = dlarray(chol_input);  % shape: [1, batch]

        % Replace cholesterol row in input (so it's the only diff'd input)
        mod_inputs = ins;
        mod_inputs(chol_index,:) = chol_input;

        % Define scalar function: sum or mean of predictions
        f = @() sum(forward(network, mod_inputs), 'all');

        % Compute gradient of sum of y_pred w.r.t. cholesterol feature (vectorized)
        grad = dlgradient(f(), chol_input, 'EnableHigherDerivatives', true);  % returns [1, batch]

        % Set constraint value, dl format
        cI_dl(i) = max(-grad(:));

        % Set constraint value
        cI(i,1) = extractdata(max(-grad(:)));
    
        % Compute constraint gradient and add to Jacobian
        grad = dlgradient(cI_dl(i), network.Learnables);
        JI(i,:) = double(self.flattenGradient(grad))';

      end

    end % evaluateConstraintFunctionAndJacobianInequalities

    % Evaluate objective function and gradient value
    function [f,g] = evaluateObjectiveFunctionAndGradient(self,network,inputs,outputs)

      % Evaluate forward passes
      y_pred = forward(network,inputs);

      % Evaluate logistic loss (maps {0,1} values to {-1,1} values)
      f = sum(log(1 + exp(-(2*y_pred-1).*(2*outputs-1))))/length(y_pred);

      % Evaluate deep learning objective gradient
      g = dlgradient(f,network.Learnables);
      
      % Set function value
      f = double(extractdata(f));

      % Set gradient value
      g = double(self.flattenGradient(g));

    end % evaluateObjectiveFunctionAndGradient

    % Flatten gradient
    function g = flattenGradient(self,g)

      % Reshape values
      for i = 1:self.n_hidden+1
        g_W{i} = reshape(extractdata(cell2mat(g.Value(2*(i-1)+1))), [], 1);
        g_b{i} = reshape(extractdata(cell2mat(g.Value(2*(i-1)+2))), [], 1);
      end

      % Vectorize
      g = [];
      for i = 1:self.n_hidden+1
        g = [g; g_W{i}; g_b{i}];
      end

    end % flattenGradient

    % Initialize neural network
    function network = initializeNeuralNetwork(self,W,b)

      % Set input layer
      layers = [featureInputLayer(self.n_features)];

      % Loop through hidden layers
      for i = 1:self.n_hidden
        layers = [layers
                  fullyConnectedLayer(self.n_hidden_nodes(i),'Name','layer','Weights',W{i},'Bias',b{i})
                  tanhLayer];
      end

      % Set output layer
      layers = [layers
                fullyConnectedLayer(self.n_outputs,'Name','layer','Weights',W{self.n_hidden+1},'Bias',b{self.n_hidden+1})];
      
      % Finalize network
      network = dlnetwork(layers);

    end % initializeNeuralNetwork

    % Number of variables
    function n = numberOfVariables(self)

      % Number of variables
      n = self.n_variables;

    end % numberOfVariables

    % Objective function and gradient
    function [f,g] = objectiveFunctionAndGradient(self,w,option)

      % Update network variables
      self.updateNetworkVariables(w);

      % Check type
      if option == 0 || size(self.inputs_obj,2) == 0
        ins  = self.inputs_obj;
        outs = self.outputs_obj;
      else
        indices = round(1 + (size(self.inputs_obj,2)-1)*rand(1,self.batch_size));
        ins     = dlarray(self.inputs_obj(:,indices),'CB');
        outs    = dlarray(self.outputs_obj(:,indices),'CB');
      end
      
      % Evaluate mse function
      [f,g] = dlfeval(@self.evaluateObjectiveFunctionAndGradient,self.network,ins,outs);
      
    end % objectiveFunction

    % Predictions
    function y_pred = predictions(self,inputs,w)

      % Set test inputs
      inputs = dlarray(inputs,'CB');

      % Set best iterate
      self.updateNetworkVariables(w);

      % Evaluate forward passes
      y_pred = forward(self.network, inputs);

      % Extract data
      y_pred = extractdata(y_pred);

    end % predictions
    
    % Update network variables
    function updateNetworkVariables(self,w)

      % Check if equal to last
      if isequal(w,self.w_last) 
        return
      end

      % Set indices
      if self.n_hidden == 0
        W_idx{1} = 1:(self.n_features * self.n_outputs);
        b_idx{1} = W_idx{1}(end) + [1:self.n_outputs];
      else
        W_idx{1} = 1:(self.n_features * self.n_hidden_nodes(1));
        b_idx{1} = W_idx{1}(end) + [1:self.n_hidden_nodes(1)];
        for i = 2:self.n_hidden
          W_idx{i} = b_idx{i-1}(end) + [1:(self.n_hidden_nodes(i-1)*self.n_hidden_nodes(i))];
          b_idx{i} = W_idx{i}(end) + [1:self.n_hidden_nodes(i)];
        end
        W_idx{self.n_hidden+1} = b_idx{self.n_hidden}(end) + [1:(self.n_hidden_nodes(end)*self.n_outputs)];
        b_idx{self.n_hidden+1} = W_idx{self.n_hidden+1}(end) + [1:self.n_outputs];
      end

      % Separate variables
      if self.n_hidden == 0
        W{1} = reshape(w(W_idx{1}), [self.n_outputs, self.n_features]);
        b{1} = reshape(w(b_idx{1}), [self.n_outputs, 1]);
      else
        W{1} = reshape(w(W_idx{1}), [self.n_hidden_nodes(1), self.n_features]);
        b{1} = reshape(w(b_idx{1}), [self.n_hidden_nodes(1), 1]);
        for i = 2:self.n_hidden
          W{i} = reshape(w(W_idx{i}), [self.n_hidden_nodes(i), self.n_hidden_nodes(i-1)]);
          b{i} = reshape(w(b_idx{i}), [self.n_hidden_nodes(i), 1]);
        end
        W{self.n_hidden+1} = reshape(w(W_idx{self.n_hidden+1}), [self.n_outputs, self.n_hidden_nodes(end)]);
        b{self.n_hidden+1} = reshape(w(b_idx{self.n_hidden+1}), [self.n_outputs, 1]);
      end

      % Set variable table
      for i = 1:self.n_hidden+1
        self.w_table.Value{2*(i-1)+1,1} = dlarray(W{i});
        self.w_table.Value{2*(i-1)+2,1} = dlarray(b{i});
      end

      % Update network weights
      updateFunction = @(network,values) values;
      self.network   = dlupdate(updateFunction, self.network, self.w_table);
      self.w_last    = w;

    end % updateNetworkVariables

  end % methods

end % constrainedClassification
