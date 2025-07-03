function [sol, result] = solve_LP_LPEC_gurobi(c, grad_c, grad_f, K1)

    % Dimensions
    m = length(c);           % number of constraints
    n = length(grad_f);      % number of variables

    % Indices
    idx_neg = c < 0;
    idx_pos = c >= 0;

    %-------------------------%
    % Build Gurobi model
    %-------------------------%
    model = struct();
    model.modelsense = 'min';

    % Decision variables: [lambda; u; v]
    % Total variables: m + n + n
    num_vars = m + 2*n;

    % Objective function
    obj = zeros(num_vars, 1);
    obj(1:m) = -c .* idx_neg;     % -c_i for c_i < 0
    obj(m+1:m+n) = 1;             % eᵀu
    obj(m+n+1:end) = 1;           % eᵀv

    model.obj = obj;

    % Equality constraint: grad_c^T * lambda + grad_f = u - v
    % Aeq * x = -grad_f
    Aeq = zeros(n, num_vars);
    Aeq(:, 1:m) = grad_c';           % grad_c^T * lambda
    Aeq(:, m+1:m+n) = -eye(n);       % -u
    Aeq(:, m+n+1:end) = eye(n);      % +v

    model.A = sparse(Aeq);
    model.rhs = -grad_f;
    model.sense = repmat('=', n, 1);

    % Bounds
    lb = zeros(num_vars, 1);
    ub = inf(num_vars, 1);
    ub(1:m) = K1;  % upper bound for lambda

    model.lb = lb;
    model.ub = ub;

    % Gurobi parameters
    params.OutputFlag = 0;

    % Solve
    result = gurobi(model, params);

    if ~strcmp(result.status, 'OPTIMAL')
        error('LP solve failed: %s', result.status);
    end

    sol.z = result.x(1:m);
    sol.u = result.x(m+1:m+n);
    sol.v = result.x(m+n+1:end);

end
