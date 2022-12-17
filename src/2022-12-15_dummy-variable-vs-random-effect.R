# https://stats.stackexchange.com/questions/507518/is-a-random-intercept-model-exactly-the-same-as-a-linear-model-with-dummy-variab

library(lme4)
library(magrittr)
library(dplyr)

x_range <- c(150, 400)
y_range <- c(0, 10) 


# Model 0: Using dummy variables for group 
m0_dummy <- lm(Reaction ~ Subject, data = sleepstudy)
summary(m0_dummy)

df_m0_coef <- m0_dummy$coefficients %>% as.data.frame()
baseline_intercept <- df_m0_coef[1,1]
df_m0_coef <- df_m0_coef %>% 
  mutate(group_intercept = ifelse(. != baseline_intercept,
                                  baseline_intercept + .,
                                  .)) %>% 
  select(group_intercept)


# Model 1: Using random effect for group 
m1_random_effects <- lmer(Reaction ~ 1 + (1 | Subject), data = sleepstudy)
summary(m1_random_effects)

df_m1_coef <- coef(m1_random_effects)$Subject


# Plots 
par(mfrow = c(1, 2))

df_m0_coef$group_intercept %>% 
  hist(xlim = x_range, ylim = y_range, breaks = 10, 
       main = 'Dummy model: \nEstimates of group intercepts')

df_m1_coef$`(Intercept)` %>% 
  hist(xlim = x_range, ylim = y_range, breaks = 10, 
       main = 'Random effects model: \nEstimates of group intercepts')
