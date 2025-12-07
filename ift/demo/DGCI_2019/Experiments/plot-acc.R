# pacotes necessários
# comandos para instala-los 
# install.packages("readr")
# install.packages("dplyr")
# install.packages("ggplot2")

library(readr)
library(dplyr)
library(ggplot2)

# arquivo com os dados
df <- read_csv("exp-data-dgci.csv")

#######################
#  INICIO PARAMETROS  #
#######################

# "000001_000002" "000001_000003" "000001_000004" "000001_000005" "000001_000006"
# "000001_000018" "000001_000020" "000001_000024" "000001_000026" "000001_000027"
# "000001_000028" "000001_000029" "000001_000030" "000001_000032" "000001_000034"
# "000001_000035" "000001_000038" "000001_000039" "000001_000040" "000001_000041"
# "000001_000043" "000001_000044" "000001_000046" "000001_000048" "000001_000050"

# imagem selecionada
img_name <- '000001_000005'

# extensão do output
ext <- '.pdf'

#  "GCmax-w1"    "GCmax-w2"     "GCsum-w3"     "GCsum-w4"
#  "GCmax-DT-w5" "GCmax-DT-w6"  "GCmax-DT-w7"  "GCmax-DT-w8"
#  "GCmax-DT-w9" "GCmax-DT-w10" "GCmax-DT-w11" "GCmax-DT-w12"

# métodos selecionados
methods <- c('GCmax-w1', 'GCsum-w3', 'GCmax-DT-w9', 'GCmax-DT-w10')

# intervalo das iterações que irão aparecer no plot
min_ite <- 5
max_ite <- 15

# intervalo da acurácia que ira aparecer no plot
min_acc <- 0.5
max_acc <- 1.0

####################
#  FIM PARAMETROS  #
####################

# código para gerar o plot
p <- df %>% filter(image == img_name) %>% filter(method %in% methods) %>% 
    ggplot(aes(x = ite, y = acc, group = method, color = method)) +
    geom_line() + geom_point() + xlim(min_ite, max_ite) + ylim(min_acc, max_acc) +
    labs(title = paste("Accuracy per Iteration on Image", img_name), 
         color = "Method", x = "Iteration", y = "Accuracy")

# salvando o plot
ggsave(filename=paste0('plot-', img_name, ext), plot=p, width=6, height=3)