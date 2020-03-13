## ---- generate_data

#library(cmdstanr)
library(rstan)
library(tidybayes)

tempenv <- new.env(parent = baseenv())
source('data_for_generation.R', local = tempenv)
gen_data <- as.list(tempenv)

# Paths may be different for you. For me compiled stan code in cmdstan directory
# ../cmdstan/mlmed2/

# From there, run (change num_samples if you want more simulations): 
# ./mlmed_example sample algorithm=hmc engine=nuts max_depth=15 num_samples=8 num_warmup=500 output file=../../mlmed_example/sim.csv data file=../../mlmed_example/data_for_generation.R

gend <- rstan::read_stan_csv('sim.csv')
ysim_names <- grep('Y_sim', names(gend), value = T)
msim_names <- grep('M_sim', names(gend), value = T)
y_sim <- as.array(gend, pars = ysim_names)
m_sim <- as.array(gend, pars = msim_names)

to_fit_data <- gen_data
to_fit_data$SIMULATE <- 0
to_fit_data$prior_bs <- 1            #normal(0,1)
to_fit_data$prior_mbeta <- 1         #normal(0,1)
to_fit_data$prior_ybeta <- 1         #normal(0,1)
to_fit_data$prior_sigmas <- 1        #exponential(0,1)
to_fit_data$prior_id_lkj_shape <- 1  #lkj_corr_cholesky(1)
to_fit_data$prior_roi_lkj_shape <- 1 #lkj_corr_cholesky(1)
to_fit_data$prior_id_taus <- 2.5     #cauchy(0,2.5)
to_fit_data$prior_roi_taus <- 2.5    #cauchy(0,2.5)

invisible(lapply(1:dim(y_sim)[1], function(i){
  temp <- to_fit_data
  temp$Y <- y_sim[i, 1, ]
  temp$M <- m_sim[i, 1, ]  
  stan_rdump(ls(temp), sprintf('simdata%02d.R', i), envir = list2env(temp))
  return(NULL)
}))

#bash cmdstan_chain_per_sim.bash

