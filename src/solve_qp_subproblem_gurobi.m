function [delta_x, result] = solve_qp_subproblem_gurobi(H, grad_f, grad_c, c, lb, ub, x)

    % Solve the QP subproblem using Gurobi
    model = struct();
    model.modelsense = 'min';
    model.Q = sparse(H);  % Hessian matrix
    model.obj = grad_f;  % Gradient of the objective function
    model.A = sparse(grad_c);  % Gradient of the constraints
    model.rhs = -c;  % Right-hand side of the constraints
    model.sense = repmat('<', length(c), 1);  % Sense of the constraints (<=)
    model.lb = lb - x;  % Lower bounds on the variables
    model.ub = ub - x;  % Upper bounds on the variables
    
    % Set Gurobi parameters
    params = struct();
    params.Method = 0; % Use dual simplex method
    params.OutputFlag = 0;  % Suppress output
    params.FeasibilityTol = 1e-8;
    params.OptimalityTol = 1e-8;
    
    % Solve the QP problem
    result = gurobi(model, params);
    % Check method used...
    
    if ~strcmp(result.status, 'OPTIMAL')
        error('QP subproblem failed: %s', result.status);
    end
    
    % Return the solution to the QP subproblem
    delta_x = result.x;
    
end
