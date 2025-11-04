function [active_set,cJd] = estimateActiveSetQP(g,cE,JE,cI,JI,theta,nu,QP_tol)

% Description : Estimates active set with QP approach
% Authors     : Lara Zebiane and Frank E. Curtis
% Inputs      : see paper
% Outputs     : active_set, estimated active indices
%               threshold, threshold in active-set estimation for printing

% Set number of variables
n = length(g);

% Set numbers of equality and inequality constraints
mE = length(cE);
mI = length(cI);

% Set indicator for solving primal or dual
if (mE > 0 && norm((1/theta)*JE*JE') <= 1e-08) || (mI > 0 && norm((1/theta)*JI*JI') <= 1e-08)
  primal = true;
else
  primal = false;
end

% Check indicator
if primal

  % Construct objective and bounds
  g_QP = [g; nu*ones(2*mE+mI,1)];
  Q_QP = sparse(n+2*mE+mI,n+2*mE+mI);
  for i = 1:n
    Q_QP(i,i) = theta;
  end
  l_QP = [-inf*ones(n,1); zeros(2*mE+mI,1)];
  u_QP = inf*ones(n+2*mE+mI,1);

  % Inequality constraint data
  A_QP = [JI sparse(mI,2*mE) -speye(mI)];
  b_QP = -cI;

  % Equality constraint data
  AE_QP = [JE -speye(mE) speye(mE) sparse(mE,mI)];
  bE_QP = -cE;

  % Suppress output
  options = optimoptions('quadprog','Display','none');

  % Solve QP
  sol = quadprog(Q_QP,g_QP,A_QP,b_QP,AE_QP,bE_QP,l_QP,u_QP,[],options);

  % Parse solution
  d = sol(1:n);

else % solve dual!

  % Construct objective and bounds
  g_QP = -[cE; cI];
  if mE > 0
    g_QP(1:mE) = g_QP(1:mE) + (1/theta)*JE*g;
  end
  if mI > 0
    g_QP(mE+1:mE+mI) = g_QP(mE+1:mE+mI) + (1/theta)*JI*g;
  end
  if mE > 0 && mI == 0
    Q_QP = (1/theta)*JE*JE';
  elseif mE == 0 && mI > 0
    Q_QP = (1/theta)*JI*JI';
  else
    Q_QP = (1/theta)*[JE*JE' JE*JI'; JI*JE' JI*JI'];
  end
  Q_QP = (1/2)*(Q_QP + Q_QP');
  l_QP = [-nu*ones(mE,1);   zeros(mI,1)];
  u_QP = [ nu*ones(mE,1); nu*ones(mI,1)];

  % Suppress output
  options = optimoptions('quadprog','Display','none');

  % Solve QP
  sol = quadprog(Q_QP,g_QP,[],[],[],[],l_QP,u_QP,[],options);

  % Parse solution
  alpha = sol(1:mE);
  beta  = sol(mE+1:mE+mI);

  % Set primal solution
  d = -g;
  if mE > 0
    d = d - JE'*alpha;
  end
  if mI > 0
    d = d - JI'*beta;
  end
  d = (1/theta)*d;

end

% Compute linearized constraint value
cJd = cI + JI*d;

% Estimate active set
active_set = [];
for i = 1:mI
  if cJd(i) >= -QP_tol
    active_set = [active_set i];
  end
end