setwd('C:/Users/akwn2/Documents/stipple/cases')
source("hmc.r")

# Start timer
t0 = proc.time()

#################################################################
## Data generation
#################################################################

# True parameters
N <- 10
w1 <- 5.
w2 <- 1.

# Generating inputs and outputs
xin <- seq(0, 10, length=N)
theta <- 1 / (1 + exp( -(w1 * xin + w2) ))
yout = c()
for (ii in 1:N) {
  yout <- c(yout, rbinom(1,1, theta[ii]))
}

#################################################################
## Energy function and gradient
#################################################################

# Modelling function
energy <- function(var, x, y){
  
  # Parameters
  N <- 10
  mu_w1 <- 1.
  s2_w1 <- 10.
  mu_w2 <- 5.
  s2_w2 <- 10.
  
  # Variables
  w1 <- var[1]
  w2 <- var[2]
  
  # Hyper-priors, non-informative
  pri1 <- - 0.5 * log(2 * pi * s2_w1) - 0.5 * (w1 - mu_w1) ^ 2 / s2_w1
  pri2 <- - 0.5 * log(2 * pi * s2_w2) - 0.5 * (w2 - mu_w2) ^ 2 / s2_w2
  
  # Likelihood term
  theta <- 1 / (1 + exp( -(w1 * x + w2)) )
  like <- sum( y * log(theta) + (1 - y) * log(1 - theta) )
  
  # Log joint
  return (like + pri1 + pri2)
}

grad_energy <- function(var, x, y){
  
  # Parameters
  N <- 10
  mu_w1 <- 1.
  s2_w1 <- 10.
  mu_w2 <- 5.
  s2_w2 <- 10.
  
  # Variables
  w1 = var[1]
  w2 = var[2]
  theta <- 1 / (1 + exp( -(w1 * x + w2)) )
  
  # Hyper-priors, non-informative
  d_w1 <- - (w1 - mu_w1) / s2_w1
  d_w2 <- - (w2 - mu_w2) / s2_w2
  
  # Likelihood term
  dlik_theta = y / theta - (1 - y) / (1 - theta)
  d_theta_w1 = x * exp( -(w1 * x + w2)) / (1 + exp( -(w1 * x + w2)) ) ^ 2
  d_theta_w2 = exp( -(w1 * x + w2)) / (1 + exp( -(w1 * x + w2)) ) ^ 2
  
  d_w1 <- d_w1 + sum( dlik_theta * d_theta_w1 )
  d_w2 <- d_w2 + sum( dlik_theta * d_theta_w2 )

  # Log joint
  return ( c(d_w1, d_w2) )
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
x0 = c(1.0, 1.0)       # initial state
step_size = 0.0001     # size of leapfrog steps
n_steps = 5            # leapfrog steps

samples = c()
while (n_samples < total_samples && rejections < max_rejections){
  
  x = HMC(f, g, step_size, n_steps, x0)  
  
  if (x != x0){
    n_samples = n_samples + 1
    samples <- c(samples, x)
    x0 = x
  }
  else{
    rejections = rejections + 1
  }
}

# End timer
tf = proc.time()
elapsed_time = tf - t0
