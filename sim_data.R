library(cmdstanr)
library(rstan)
library(tidybayes)

tempenv <- new.env(parent = baseenv())
source('../cmdstan/mlmed/data_for_generation.R', local = tempenv)
gen_data <- as.list(tempenv)

#./mlmed_example sample algorithm=hmc engine=nuts max_depth=15 num_samples=1 num_warmup=500 output file=sim.csv data file=data_for_generation.R

gend <- rstan::read_stan_csv('../cmdstan/mlmed/sim.csv')
ysim_names <- grep('Y_sim', names(gend), value = T)
msim_names <- grep('M_sim', names(gend), value = T)
y_sim <- unlist(lapply(rstan::extract(gend, pars = ysim_names), `[[`, 1))
m_sim <- unlist(lapply(rstan::extract(gend, pars = msim_names), `[[`, 1))

to_fit_data <- gen_data
to_fit_data$Y <- y_sim
to_fit_data$M <- m_sim
to_fit_data$SIMULATE <- 0
to_fit_data$prior_bs <- 1            #normal(0,1)
to_fit_data$prior_mbeta <- 1         #normal(0,1)
to_fit_data$prior_ybeta <- 1         #normal(0,1)
to_fit_data$prior_sigmas <- 1        #exponential(0,1)
to_fit_data$prior_id_lkj_shape <- 1  #lkj_corr_cholesky(1)
to_fit_data$prior_roi_lkj_shape <- 1 #lkj_corr_cholesky(1)
to_fit_data$prior_id_taus <- 2.5     #cauchy(0,2.5)
to_fit_data$prior_roi_taus <- 2.5    #cauchy(0,2.5)

stan_rdump(ls(to_fit_data), '../cmdstan/mlmed/data_to_fit.R', envir = list2env(to_fit_data))

#time ./mlmed_example sample algorithm=hmc engine=nuts max_depth=15 num_samples=1000 num_warmup=1000 output file=fit.csv data file=data_to_fit.R

fit <- rstan::read_stan_csv('../cmdstan/mlmed/fit.csv')

params <- tidybayes::gather_draws(gend, `(.*_(a|b|cp|dy|dm|ty|tm).*|(a|b|cp|dy|dm|ty|tm))`, regex = TRUE)
cor_params <- tidybayes::gather_draws(gend, `.*Omega.*`, regex = TRUE)
tau_params <- tidybayes::gather_draws(gend, `.*Tau.*`, regex = TRUE)

params_from_sim <- tidybayes::gather_draws(fit, `(.*_(a|b|cp|dy|dm|ty|tm).*|(a|b|cp|dy|dm|ty|tm))`, regex = TRUE)

cor_params_from_sim <- tidybayes::gather_draws(fit, `.*Omega.*`, regex = TRUE)
tau_params_from_sim <- tidybayes::gather_draws(fit, `.*Tau.*`, regex = TRUE)

v_pars <- params_from_sim %>% group_by(.variable) %>%
    summarize(mean(.value)) %>%
    left_join(params[params$.iteration == 1,]) %>%
    filter(grepl('v_', .variable)) %>%
    extract(.variable, c('var', 'var2', 'idx'), regex = '(v_)(.*)\\.(.*)')

v_pars %>%
    ggplot(aes(x = .value, y = `mean(.value)`)) +
    geom_point() +
    geom_smooth(method = 'lm') +
    facet_wrap(~var2, scales = 'free') +
    geom_abline(intercept = 0, slope = 1)

qplot(v_pars$.value[v_pars$var2 == 'dm'], v_pars$`mean(.value)`[v_pars$var2 == 'b'],
      geom = c('point', 'smooth'), method = 'lm') +
    geom_abline(intercept = 0, slope = 1)

qplot(v_pars$.value[v_pars$var2 == 'tm'], v_pars$`mean(.value)`[v_pars$var2 == 'b'],
      geom = c('point', 'smooth'), method = 'lm') +
    geom_abline(intercept = 0, slope = 1)


params_from_sim %>% group_by(.variable) %>%
    summarize(mean(.value)) %>%
    left_join(params[params$.iteration == 1,]) %>%
    filter(grepl('^u_', .variable)) %>%
    extract(.variable, c('var', 'var2', 'idx'), regex = '(u_)(.*)\\.(.*)') %>%
    ggplot(aes(x = .value, y = `mean(.value)`)) +
    geom_point() +
    geom_smooth(method = 'lm') +
    facet_wrap(~var2, scales = 'free') +
    geom_abline(intercept = 0, slope = 1)

params_from_sim %>% group_by(.variable) %>%
    summarize(mean(.value)) %>%
    left_join(params[params$.iteration == 1,]) %>%
    filter(!grepl('^(v|u)_', .variable)) %>%
    ggplot(aes(x = .value, y = `mean(.value)`)) +
    geom_point() +
    geom_smooth(method = 'lm')

params_from_sim %>% group_by(.variable) %>%
    summarize(mean(.value)) %>%
    left_join(params[params$.iteration == 1,]) %>%
    filter(grepl('tau', .variable)) %>%
    ggplot(aes(x = .value, y = `mean(.value)`)) +
    geom_point() +
    geom_smooth(method = 'lm')

cor_params_from_sim %>% group_by(.variable) %>%
    summarize(mean(.value)) %>%
    left_join(cor_params[cor_params$.iteration == 1,]) %>% View

tau_params_from_sim %>% group_by(.variable) %>%
    summarize(mean(.value)) %>%
    left_join(tau_params[tau_params$.iteration == 1,]) %>% View
