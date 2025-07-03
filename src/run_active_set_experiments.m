function run_active_set_experiments()
    subsolvers = {'QP', 'LPLPEC'};  % The solvers we want to compare
    problems = {'problem1', 'problem2'}; % Problems we're solving
    noise_levels = [0; 1e-5; 1e-2; 1e-1];  % Different levels of random noise to test
    % noise_levels = [1e-2];  % Different levels of random noise to test

    for p = 1:length(problems)
        problem = problems{p};
        for k = 1:length(subsolvers)
            subsolver = subsolvers{k};
            for i = 1:length(noise_levels)
                noise = noise_levels(i);

                % If there's no noise, one trial is enough (deterministic)
                num_trials = (noise == 0) * 1 + (noise ~= 0) * 5;

                % Call the function that runs the actual experiment
                run_single_experiment(subsolver, noise, num_trials, problem);
            end
        end
    end
end


function run_single_experiment(subsolver, noise_level, num_trials, problem)
    tol = 2e-2;
    rho = 20;
    K1 = 2;
    sigma = 0.75;
    beta = 0.7071;

    x1_vals = -1:0.005:-0.3; 
    x2_vals = -0.25:0.005:0.75;
    n1 = length(x1_vals);
    n2 = length(x2_vals);
    active_set_map = zeros(n2, n1);

    for i = 1:n1
        for j = 1:n2
            x = [x1_vals(i); x2_vals(j)];

            % === Select objective based on problem ===
            if strcmp(problem, 'problem1')
                [~, grad_f] = objective1(x);
            elseif strcmp(problem, 'problem2')
                [~, grad_f] = objective2(x);
            else
                error('Unknown problem: %s', problem);
            end

            [c, grad_c] = constraints(x);
            success_count = 0;

            for trial = 1:num_trials
                grad_f_perturbed = grad_f + noise_level * (2 * rand(size(grad_f)) - 1);
                c_perturbed = c + noise_level * (2 * rand(size(c)) - 1);
                grad_c_perturbed = grad_c + noise_level * (2 * rand(size(grad_c)) - 1);

                try
                    if strcmp(subsolver, 'QP')
                        H = rho * eye(length(x));
                        lb = [-inf; -inf]; ub = [inf; inf];
                        [delta_x, ~] = solve_qp_subproblem_gurobi(H, grad_f_perturbed, grad_c_perturbed, c_perturbed, lb, ub, x);
                        residual = c_perturbed + grad_c_perturbed * delta_x;
                        if strcmp(problem, 'problem1')
                            success = residual(1) < -tol && residual(2) >= -tol;
                        else
                            success = residual(1) >= -tol && residual(2) >= -tol;
                        end

                    elseif strcmp(subsolver, 'LPLPEC')
                        [sol, ~] = solve_LP_LPEC_gurobi(c_perturbed, grad_c_perturbed, grad_f_perturbed, K1);
                        rho_bar = compute_rho_bar(c_perturbed, sol.z, grad_c_perturbed, grad_f_perturbed);
                        residual = c_perturbed + (beta * rho_bar)^sigma;
                        
                        if strcmp(problem, 'problem1')
                            success = residual(1) < -tol && residual(2) >= -tol;
                        else
                            success = residual(1) >= -tol && residual(2) >= -tol;
                        end

                    else
                        error('Unknown subsolver: %s', subsolver);
                    end
                catch
                    success = false;
                end

                if success
                    success_count = success_count + 1;
                end
            end

            active_set_map(j, i) = success_count / num_trials;

            fprintf('[%s | %s] x = [%.3f, %.3f], Successes = %d/%d\n', ...
                    subsolver, problem, x(1), x(2), success_count, num_trials);
        end
    end

    % === Plot ===
    figure;
    imagesc(x1_vals, x2_vals, active_set_map);
    set(gca, 'YDir', 'normal');
    colormap(parula);
    colorbar;
    caxis([0 1]);
    axis equal tight;
    xlabel('x₁');
    ylabel('x₂');
    title(sprintf('activeset\\_%s\\_%s\\_eps\\_%.1e\\_tol\\_%.1e\\_trials\\_%d', ...
        problem, subsolver, noise_level, tol, num_trials));

    % Constraint boundaries
    hold on;
    x1_parabola = linspace(min(x1_vals), max(x1_vals), 300);
    x2_parabola1 = x1_parabola.^2;
    x2_parabola2 = 0.5 - x1_parabola.^2;
    in_bounds1 = (x2_parabola1 >= min(x2_vals)) & (x2_parabola1 <= max(x2_vals));
    in_bounds2 = (x2_parabola2 >= min(x2_vals)) & (x2_parabola2 <= max(x2_vals));
    plot(x1_parabola(in_bounds1), x2_parabola1(in_bounds1), 'k-', 'LineWidth', 1);
    plot(x1_parabola(in_bounds2), x2_parabola2(in_bounds2), 'k-', 'LineWidth', 1);
    hold off;

    % === Save Figure ===
    [~, ~] = mkdir('figures');
    filename = sprintf('figures/activeset_%s_%s_eps_%.0e_tol_%.0e_trials_%d.fig', ...
                problem, subsolver, noise_level, tol, num_trials);
    saveas(gcf, replace(filename, '.fig', '.png'));
    savefig(filename);
end


function [f, grad_f] = objective1(x)
    % f(x) = (x1 + 0.5)^2 + (x2 - 0.5)^2
    f = (x(1) + 0.5)^2 + 4 * (x(2) - 0.5)^2;
    grad_f = [2 * x(1) + 1;
              8 * x(2) - 4];
end

function [f, grad_f] = objective2(x)
    % f(x) = 4*(x1 + 0.5)^2 + (x2 - 0.25)^2
    f = 4 * (x(1) + 0.6)^2 + (x(2) - 0.25)^2;
    grad_f = [8 * x(1) + 4.8;
              2 * x(2) - 0.5];
end

function [c, grad_c] = constraints(x)
    % Two inequality constraints: x1^2 - x2 <= 0 and x1^2 + x2 - 0.5 <= 0
    c = [x(1)^2 - x(2);
         x(1)^2 + x(2) - 0.5];  % this is c2, the active constraint for objective2

    % Jacobian (gradient) of each constraint
    grad_c = [2 * x(1), -1;
              2 * x(1),  1];
end

function rho_bar = compute_rho_bar(c, z, grad_c, grad_f)
    % This function calculates rho_bar for LP_LPEC

    % First term: for constraints that are inactive (c < 0)
    mask_neg = c < 0;
    term1 = sum(sqrt(-c(mask_neg) .* z(mask_neg)));

    % Second term: for constraints that are already satisfied (c >= 0)
    mask_pos = c >= 0;
    term2 = sum(c(mask_pos));

    % Third term: stationarity residual
    term3 = norm(grad_c' * z + grad_f);

    % Final rho_bar value
    rho_bar = term1 + term2 + term3;
end

