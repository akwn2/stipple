function samples = benchModel1Matlab()
    % Start timer
    tic();

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Data generation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % True parameters
    N = 10;
    w1 = 5.;
    w2 = 1.;

    mu_w1 = 1.;
    s2_w1 = 10.;
    mu_w2 = 5.;
    s2_w2 = 10.;

    % Generating inputs and outputs
    xin = linspace(0, 10, N)';
    theta = 1 ./ (1 + exp( - (w1 .* xin + w2 ) ) );
    yout = zeros(N, 1);
    
    for ii = 1:N
        yout(ii) = rand() >= theta(ii);
    end

    f = @(x) energy(x, xin, yout);
    g = @(x) grad_energy(x, xin, yout);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Main HMC execution
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % sampler parameters
    n_samples = 0;
    max_rejections = 5000;
    total_samples = 5000;
    rejections = 0;

    % sampler tuning
    x0 = [1.0, 1.0]';  % initial state
    step_size = 0.0001;     % size of leapfrog steps
    n_steps = 5;            % leapfrog steps
    samples = [];
    
    while (n_samples < total_samples && rejections < max_rejections)

      x = HMC(f, g, step_size, n_steps, x0);

      if (x ~= x0)
        n_samples = n_samples + 1;
        samples = [samples, x];
        x0 = x;
      else
        rejections = rejections + 1;
      end
    end

    % End timer
    elapsed_time = toc();
    
    fprintf('Total elapsed time: %e\n', elapsed_time)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Energy function and gradient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Modelling function
function f_val = energy(var, x, y)
% Parameters
    N = 10;
    mu_w1 = 1.;
    s2_w1 = 10.;
    mu_w2 = 5.;
    s2_w2 = 10.;

    % Variables
    w1 = var(1);
    w2 = var(2);
    theta = 1 ./ (1 + exp( - (w1 .* x + w2) ) );
    
    % Hyper-priors, non-informative
    pri1 = - 0.5 .* log(2 .* pi .* s2_w1) - 0.5 .* (w1 - mu_w1) ^ 2 ./ s2_w1;
    pri2 = - 0.5 .* log(2 .* pi .* s2_w2) - 0.5 .* (w2 - mu_w2) ^ 2 ./ s2_w2;

    % Likelihood term
    like = sum(y .* log(theta) + (1 - y) .* log(1 - theta) );

    % Log joint
    f_val = like + pri1 + pri2;
end                    
                    

function df = grad_energy(var, x, y)
    % Parameters
    N = 10;
    w1 = 5.;
    w2 = 1.;
    s2 = 0.1;
    a = 0.01;
    b = 0.01;
    mu_w1 = 1.;
    s2_w1 = 10.;
    mu_w2 = 5.;
    s2_w2 = 10.;

    % Variables
    w1 = var(1);
    w2 = var(2);
    theta = 1 ./ (1 + exp( - (w1 .* x + w2) ) );

    % Hyper-priors, non-informative
    d_w1 = - (w1 - mu_w1) / s2_w1;
    d_w2 = - (w2 - mu_w2) / s2_w2;

    % Likelihood term
    dlik_theta = y ./ theta - (1 - y) ./ (1 - theta);
    d_theta_w1 = x .* exp( -(w1 * x + w2)) ./ ((1 + exp( -(w1 * x + w2)) ) .^ 2);
    d_theta_w2 = exp( -(w1 * x + w2)) ./ ((1 + exp( -(w1 * x + w2)) ) .^ 2);
  
    d_w1 = d_w1 + sum( dlik_theta(:) .* d_theta_w1(:) );
    d_w2 = d_w2 + sum( dlik_theta(:) .* d_theta_w2(:) );
    
    df = zeros(2,1);
    df(1) = d_w1;
    df(2) = d_w2;
 end