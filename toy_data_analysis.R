tt = read.csv('trial_type.csv')
fe = read.csv('trial_final_errors.csv')
w = read.csv('trial_magnitudes.csv')
d = data.frame(trial_type=tt$trial_type,weight=w$weight_magnitudes,final_error=fe$final_error)
d$trial_type = factor(d$trial_type)
str(d)
levels(d$trial_type) = c('easy','hard')
mod0 = lm(data=d,final_error ~ trial_type)
summary(mod0)
mod1 = lm(data=d,final_error ~ trial_type + weight)
summary(mod1)
library(ggplot2)
ggplot(data=d,aes(x=weight,y=final_error,color=trial_type)) + geom_point()

#Part 2
library(dplyr)
library(tidyr)
init_dist = read.csv('initial_cosine_distance.csv')
delta = read.csv('delta_error.csv')
d = data.frame(init_dist = init_dist$init_dist,delta=delta$delta)

cor.test(d$init_dist,d$delta)

d2 = d %>% filter(init_dist < 1.0)
cor.test(d2$init_dist,d2$delta)

ggplot(data=d,aes(x=init_dist,y=delta)) + geom_point()

sum(d$delta < 0.001) #about half show absolutely no effect

sum(d$delta)/nrow(d)
sum(d[d$init_dist < 1,]$delta)/sum(d$init_dist < 1)


init_dist = read.csv('initial_cosine_distance_2.csv')
delta = read.csv('delta_error_2.csv')
d = data.frame(init_dist = init_dist$init_dist,delta=delta$delta)

cor.test(d$init_dist,d$delta)

d2 = d %>% filter(init_dist < 1.0)
cor.test(d2$init_dist,d2$delta)

ggplot(data=d,aes(x=init_dist,y=delta)) + geom_point()

sum(d$delta < 0.001) #about half show absolutely no effect

sum(d$delta)/nrow(d)
sum(d[d$init_dist < 1,]$delta)/sum(d$init_dist < 1)
