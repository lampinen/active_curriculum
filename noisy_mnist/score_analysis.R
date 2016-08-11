library(ggplot2)
library(dplyr)
library(tidyr)

s_scores = read.csv('s_scores_ttss_10000_cs_200_batch_50_actl_4_acaat_0.950000.csv')
c_scores = read.csv('c_scores_ttss_10000_cs_200_batch_50_actl_4_acaat_0.950000.csv')
ac_scores = read.csv('ac_scores_ttss_10000_cs_200_batch_50_actl_4_acaat_0.950000.csv')

d = data.frame(accuracy = c(s_scores$accuracy,c_scores$accuracy,ac_scores$accuracy),condition=rep(c('standard','curriculum','active curriculum'),each=length(s_scores$accuracy)),trial=rep(1:length(s_scores$accuracy),3))

ggplot(data=d,aes(x=condition,y=accuracy,color=condition)) +
  geom_point() +
  theme_bw()

d_deltas = d %>% spread(condition,accuracy) %>% mutate(delta = `active curriculum`-curriculum)
ggplot(data=d_deltas,aes(delta)) +
  geom_histogram(binwidth=0.02) +
  theme_bw()


sum(ac_scores$accuracy < 0.1)
sum(c_scores$accuracy < 0.1)
sum(ac_scores$accuracy < 0.1)

sum(d_deltas$delta > 0.5)
sum(d_deltas$delta < 0)
sum(d_deltas$delta < -0.01)
