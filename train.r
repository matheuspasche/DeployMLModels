library(tidyverse)
library(tidymodels)

iris <- iris


preprocessamento <- recipe(
    Species ~.,
    data = iris )%>%
    step_normalize(all_numeric())%>%
    step_impute_median(all_numeric())


saveRDS(prep(preprocessamento,iris), "preprocessamento.rds")


modelo_knn <- nearest_neighbor(neighbors = tune())%>%
    set_engine("kknn")%>%
    set_mode("classification")%>%
    translate()

set_seed(1)

reamostra_vizinhos <- workflow()%>%
    add_recipe(preprocessamento)%>%
    add_model(modelo_knn)%>%
    tune_grid(
        resamples = mc_cv(iris, .3),
        grid = 10,
        metrics = metric_set(
            yardstick::accuracy
        )
    )


modelo_final <- workflow()%>%
    add_recipe(preprocessamento)%>%
    add_model(
        finalize_model(
            modelo_knn,
            select_best(
                reamostra_vizinhos,
                "accuracy"
            )
        )
    )%>%
    fit(iris)

print(modelo_final)

saveRDS(modelo_final, "modelo_final.rds")