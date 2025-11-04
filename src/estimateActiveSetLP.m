function [active_set,threshold] = estimateActiveSetLP(g,cE,JE,cI,JI,M,beta,sigma)

% Description : Estimates active set with LP-LPEC approach
% Authors     : Lara Zebiane and Frank E. Curtis
% Inputs      : see paper
% Outputs     : active_set, estimated active indices
%               threshold, threshold in active-set estimation for printing

% Set number of variables
n = length(g);

% Set numbers of equality and inequality constraints
mE = length(cE);
mI = length(cI);

% Construct objective and bounds
g_LP = [zeros(mE+mI,1); ones(2*n,1)];
for i = 1:mI
  if cI(i) < 0
    g_LP(mE+i) = -cI(i);
  end
end
l_LP = [-inf*ones(mE,1); zeros(mI+2*n,1)];
u_LP = M*ones(mE+mI+2*n,1);

% Equality constraint data
AE_LP = [JE' JI' -eye(n) eye(n)];
bE_LP = -g;

% Suppress output
options = optimoptions('linprog','Display','none');

% Solve LP
sol = linprog(g_LP,[],[],AE_LP,bE_LP,l_LP,u_LP,options);

% Parse solution
if ~isempty(sol)
  y = sol(1:mE);
  z = sol(mE+1:mE+mI);
else
  y = zeros(mE,1);
  z = zeros(mI,1);
end

% Evaluate rho-bar
dual = g;
if mE > 0
  dual = dual + JE'*y;
end
if mI > 0
  dual = dual + JI'*z;
end
rho_bar = norm(dual,1);
rho_bar = rho_bar + norm(cE,1);
for i = 1:mI
  if cI(i) < 0
    rho_bar = rho_bar + sqrt(-cI(i)*z(i));
  else
    rho_bar = rho_bar + cI(i);
  end
end

% Evaluate threshold
beta_scale = beta/(1 + norm(g,1) + norm(JI,1));
threshold = -(beta_scale * rho_bar)^sigma;

% Estimate active set
active_set = [];
for i = 1:mI
  if cI(i) >= threshold
    active_set = [active_set i];
  end
end