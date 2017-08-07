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
    s2 = 0.1;
    a = 0.01;
    b = 0.01;
    mu_w1 = 1.;
    s2_w1 = 10.;
    mu_w2 = 5.;
    s2_w2 = 10.;

    % Generating inputs and outputs
    xin = linspace(pi / 2., 3. * pi / 2., N)';
    noise = s2 * randn([N, 1]);
    yout = w1 .* xin + w2 .* sin(xin) + noise;

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
    x0 = [1.0, 1.0, 1.0]';  % initial state
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
    s2 = var(3);

    % Hyper-priors, non-informative
    pri1 = - 0.5 .* log(2 .* pi .* s2_w1) - 0.5 .* (w1 - mu_w1) ^ 2 ./ s2_w1;
    pri2 = - 0.5 .* log(2 .* pi .* s2_w2) - 0.5 .* (w2 - mu_w2) ^ 2 ./ s2_w2;
    pri3 = - a .* log(b) - (a + 1.) * log(s2) - b ./ s2 - gammaln(a);

    % Likelihood term
    centred =  y - (w1 .* x + w2 .* sin(x));
    loss = centred' * centred;
    like = - N * 0.5 * log(2 * pi * s2) - 0.5 * loss / s2;

    % Log joint
    f_val = like + pri1 + pri2 + pri3;
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
    s2 = var(3);

    % Hyper-priors, non-informative
    d_w1 = - (w1 - mu_w1) / s2_w1;
    d_w2 = - (w2 - mu_w2) / s2_w2;
    d_s2 = - (a + 1) / s2 + b / (s2 ^ 2);

    % Likelihood term
    centred =  y - (w1 * x + w2 * sin(x));
    loss = centred' * centred;
    d_w1 = d_w1 - sum(sum( centred ./ s2 .* x));
    d_w2 = d_w2 - sum(sum( centred ./ s2 .* sin(x)));
    d_s2 = d_s2 - N * 0.5 * log(2 * pi) / s2 + 0.5 * loss / (s2 ^ 2);

    df = zeros(3,1);
    df(1) = d_w1;
    df(2) = d_w2;
    df(3) = d_s2;
end