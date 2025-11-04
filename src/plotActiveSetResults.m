function plotActiveSetResults(option,batch_size,num_trials)

% Description : Generates plots for paper
% Authors     : Lara Zebiane and Frank E. Curtis
% Inputs      : option, 0 ~ deterministic, 1 ~ stochastic
%               batch_size, batch size
%               num_trials, number of trials per batch size
% Outputs     : png files in 'figures' directory

% Check option
if option == 0

  %%%%% DETERMINISTIC RESULTS

  % Initialize counts
  counts_LP = [];
  counts_QP = [];

  % Open LP file for reading
  filename = 'output/active_set_LP_100_0_0.out';
  file_LP = fopen(filename,'r');
  if file_LP == -1, error('Could not open file: %s', filename); end

  % Open QP file for reading
  filename = 'output/active_set_QP_100_0_0.out';
  file_QP = fopen(filename,'r');
  if file_QP == -1, error('Could not open file: %s', filename); end

  % Read each line until end-of-file
  i = 1;
  while true

    % Get line from LP file
    line_LP = fgetl(file_LP);
    if ~ischar(line_LP), break; end

    % Get line from QP file
    line_QP = fgetl(file_QP);
    if ~ischar(line_QP), break; end

    % Extract numbers from lines using regular expressions
    strings_LP = regexp(line_LP, '[-+]?\d*\.?\d+([eE][-+]?\d+)?', 'match');
    strings_QP = regexp(line_QP, '[-+]?\d*\.?\d+([eE][-+]?\d+)?', 'match');

    % Count numbers
    counts_LP(i,1) = numel(strings_LP);
    counts_QP(i,1) = numel(strings_QP);

    % Increment
    i = i + 1;

  end

  % Close files
  fclose(file_LP);
  fclose(file_QP);

  % Plot results
  figure(1);
  bar(counts_LP);
  set(gca, 'FontSize', 14);
  saveas(gcf,'figures/LP_deterministic.png');
  clf(1);
  figure(2);
  bar(counts_QP);
  set(gca, 'FontSize', 14);
  saveas(gcf,'figures/QP_deterministic.png');
  clf(2);

else

  %%%%% STOCHASTIC RESULTS

  % Initialize counts
  counts_LP = [];
  counts_QP = [];

  % Loop through repetitions
  for repetition = 1:num_trials

    % Open LP file for reading
    filename = sprintf('output/active_set_LP_100_%d_%d.out',batch_size,repetition);
    file_LP = fopen(filename,'r');
    if file_LP == -1, error('Could not open file: %s', filename); end

    % Open QP file for reading
    filename = sprintf('output/active_set_QP_100_%d_%d.out',batch_size,repetition);
    file_QP = fopen(filename,'r');
    if file_QP == -1, error('Could not open file: %s', filename); end

    % Read each line until end-of-file
    i = 1;
    while true

      % Get line from LP file
      line_LP = fgetl(file_LP);
      if ~ischar(line_LP), break; end

      % Get line from QP file
      line_QP = fgetl(file_QP);
      if ~ischar(line_QP), break; end

      % Extract numbers from lines using regular expressions
      strings_LP = regexp(line_LP, '[-+]?\d*\.?\d+([eE][-+]?\d+)?', 'match');
      strings_QP = regexp(line_QP, '[-+]?\d*\.?\d+([eE][-+]?\d+)?', 'match');

      % Count numbers
      counts_LP(i,repetition) = numel(strings_LP);
      counts_QP(i,repetition) = numel(strings_QP);

      % Increment
      i = i + 1;

    end

    % Close files
    fclose(file_LP);
    fclose(file_QP);

  end

  % Compute averages
  counts_LP = mean(counts_LP,2);
  counts_QP = mean(counts_QP,2);

  % Plot results
  figure(1);
  bar(counts_LP);
  set(gca, 'FontSize', 14);
  saveas(gcf,sprintf('figures/LP_stochastic_%d.png',batch_size));
  clf(1);
  figure(2);
  bar(counts_QP);
  set(gca, 'FontSize', 14);
  saveas(gcf,sprintf('figures/QP_stochastic_%d.png',batch_size));
  clf(2);

end