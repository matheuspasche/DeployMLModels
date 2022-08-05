library(plumber)
library(tidymodels)
library(tidyverse)

modelo_final <- readRDS('modelo_final.rds')
pre_processamento <- readRDS('preprocessamento.rds')


#' Aplica modelo
#' descricao
#' @param Sepal.Width  largura da sepala
#' @param Sepal.Length comprimento da sepala
#' @param Petal.Width  largura da petala
#' @param Petal.Length  comprimento da petala
#' @response .pred_class classe prevista
#' @post /gera_previsao
gera_previsao <- function(Sepal.Width, Sepal.Length, Petal.Width, Petal.Length){ 
    dados <- tibble(
                Sepal.Width,
                Sepal.Length,
                Petal.Width,
                Petal.Length
    )%>%
    mutate_all(as.numeric)

    prediction <- predict(
        modelo_final,
        new_data = bake(pre_processamento, dados)
    )
    bind_cols(dados, prediction)
}~

