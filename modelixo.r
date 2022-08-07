library(tidymodels)
library(tidyverse)

vb_train = read_csv('dados/Base Esteira 1 - ETAPA 1 - HSPN -  treino.csv')%>%
             select(EVER60em6, Score_customizado, SCRCRDALINSEG,SCORE_HSPN)%>%
             mutate(EVER60em6 = as.factor(EVER60em6))

vb_test = read_csv('dados/Base Esteira 1 - ETAPA 1 - HSPN -  test.csv')%>%
             select(EVER60em6, Score_customizado, SCRCRDALINSEG,SCORE_HSPN)%>%
             mutate(EVER60em6 = as.factor(EVER60em6))

vb_split <- bind_rows(vb_train, vb_test) %>% initial_split()



xgb_spec <- boost_tree(
    trees = 1000,
    tree_depth = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    mtry = tune(),
    learn_rate = tune()
)%>%
    set_engine('xgboost')%>%
    set_mode('classification')


xgb_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(), 
    finalize(mtry(),vb_train),
    learn_rate(),
    size = 20
)


xgb_wf <- workflow()%>%
    add_formula(EVER60em6 ~ .)%>%
    add_model(xgb_spec)


vb_folds <- vfold_cv(vb_train, strata = EVER60em6)

doParallel::registerDoParallel()

xgb_res <- tune_grid(
    xgb_wf,
    resamples = vb_folds,
    grid = xgb_grid,
    control = control_grid(save_pred = TRUE)
)


xgb_res%>%
    collect_metrics()%>%
    filter(.metric == 'roc_auc')%>%
    select(mean, mtry:sample_size)%>%
    pivot_longer(
        mtry:sample_size,
        names_to = 'parameter',
        values_to = 'value'
    )%>%
    ggplot(
        aes(x = value, y = mean, color = parameter)
    )+
    geom_point(show.legend = F)+
    facet_wrap(~parameter, scales = 'free_x')


show_best(xgb_res, "roc_auc")
best_auc <- select_best(xgb_res, "roc_auc")


final_xgb <- finalize_workflow(xgb_wf, best_auc)

final_xgb

library(vip)

final_xgb %>%
    fit(data = vb_train)%>%
    pull_workflow_fit()%>%
    vip(geom = 'point')


final_res <- last_fit(final_xgb, vb_split)


final_res %>%
    collect_metrics()

prediction <- final_res %>%
    collect_predictions()


 final_res %>%
    collect_predictions()%>%
    roc_curve(EVER60em6, .pred_0)%>%
    autoplot()

 prediction %>%
        rename(EVER60em6.1 = `.pred_1`)%>% 
        mutate(EVER60em6 = as.numeric(EVER60em6))%>%
        select(EVER60em6, EVER60em6.1)%>%
        filter(!is.na(EVER60em6)) %>%
        mutate(quantile = ntile(1 - EVER60em6.1, 5)) %>%
        group_by(quantile) %>%
        summarise(score_min = min(1 -EVER60em6.1) * 1000,
                         score_max = max(1 - EVER60em6.1)*1000, 
                         Mau = sum(EVER60em6), 
                         count = n(), 
                         Bom = count -Mau) %>% 
        ungroup() %>%
        mutate(Bom_count = sum(Bom),
               Mau_count = sum(Mau),
               pct_Bom = 100 * (Bom/Bom_count),
               pct_Mau = 100 * (Mau/Mau_count),
               cum_Bom = cumsum(pct_Bom), 
               cum_Mau = cumsum(pct_Mau),
               ks = cum_Mau - cum_Bom,
               EVER = Mau/count) %>%
        select(quantile, count, Mau, Bom, score_min, score_max,
               pct_Bom, pct_Mau, cum_Bom, cum_Mau, ks, EVER)


scores <- prediction%>%
    mutate(
        scoreNumerico = .pred_1 * 1000,
        score = case_when (
            between(scoreNumerico, 638, 695) ~ 'E', 
            between(scoreNumerico, 695, 749) ~ 'D', 
            between(scoreNumerico, 749, 802) ~ 'C', 
            between(scoreNumerico, 802, 866) ~ 'B', 
            between(scoreNumerico, 866, 927) ~ 'A', 
            TRUE ~ 'F'

    ))


scores %>%
    mutate(EVER60em6 = as.numeric(EVER60em6))%>%
    group_by(score)%>%
    summarise(maus = sum(EVER60em6==1))