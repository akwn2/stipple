setwd('C:/Users/akwn2/Documents/stipple/cases')
source("hmc-r.r")

# Start timer
t0 = proc.time()

#################################################################
## Data generation
#################################################################

# True parameters
N <- 3
w1 <- 30.
w2 <- 20.
s2 <- 5.

# Generating inputs and outputs
xin <- seq(pi / 2., 3. * pi / 2., length=N)
noise <- s2 * rnorm(N)
yout <- w1 * xin + w2 * sin(xin) + noise


#################################################################
## Energy function and gradient
#################################################################

# Modelling function
energy <- function(var, x, y){
  
  # Parameters
  mu_w1 <- 15.
  s2_w1 <- 100.
  mu_w2 <- 42.
  s2_w2 <- 100.
  a <- 0.001
  b <- 0.001
  N <- length(x)

    # Variables
	w1 <- var[0]
	w2 <- var[1]
	s2 <- 5.

	# Hyper-priors, non-informative
	pri1 <- - 0.5 * log(2 * pi * s2_w1) - 0.5 * (var[1] - mu_w1) ^ 2 / s2_w1
	pri2 <- - 0.5 * log(2 * pi * s2_w2) - 0.5 * (var[2] - mu_w2) ^ 2 / s2_w2
	
	# Likelihood term
	like <- 0.
	for (ii in 1:N){
		y_mu <- var[1] * x[ii] + var[2] * sin(x[ii])
		like <- like - 0.5 * log(2 * pi * s2) - 0.5 * (y[ii] - y_mu) ^ 2 / s2
	}
	# Log joint
	return (like + pri1 + pri2)
}

grad_energy <- function(var, x, y){
	w1 = var[1]
	w2 = var[2]
	
	mu_w1 = 15.
	s2_w1 = 100.
	mu_w2 = 42.
	s2_w2 = 100.

	# Hyper-priors, non-informative
	d_w1 <- - (w1 - mu_w1) / s2_w1
	d_w2 <- - (w2 - mu_w2) / s2_w2

	# Likelihood term
	like <- 0.
	for (ii in 1:N){
		y_mu <- w1 * xin[ii] + w2 * sin(xin[ii])
		d_w1 <- d_w1 - (y[ii] - y_mu) / s2 * xin[ii]
		d_w2 <- d_w2 - (y[ii] - y_mu) / s2 * sin(xin[ii])
	}
	# Log joint
	return (c(d_w1, d_w2))
}

f = function(x) energy(x, xin, yout)
g = function(x) grad_energy(x, xin, yout)



#################################################################
## Main HMC execution
#################################################################

# sampler parameters
n_samples = 0
max_rejections = 5000
total_samples = 5000
rejections = 0

# sampler tuning
x0 = c(1.0, 1.0)  # initial state
step_size = 0.001 # initial state
n_steps = 10 # initial steps
x0 = c(1.0, 1.0)  # initial state
x0 = c(1.0, 1.0)  # initial state


while (n_samples != total_samples || rejections < max_rejections){
  
  x = HMC(f, g, step_size, n_steps, x0)  
  
  if (x != x0){
    n_samples = n_samples + 1
    x0 = x
  }
  else{
    rejections = rejections + 1
  }
}

# End timer
tf = proc.time()
elapsed_time = tf - t0
