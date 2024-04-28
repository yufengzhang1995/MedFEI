library(reshape2)
library(ggplot2)

dir_root = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/gemma/task3/"

# read the results
baseline_results = read.csv(paste0(dir_root,"MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.uw_baseline_epoch5.nlg_eval_results.csv"))
enhanced2k_results = read.csv(paste0(dir_root,"MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.mimic+uw_n2000_epoch5_lr2e-4.nlg_eval_results.csv"))
enhanced3k_results = read.csv(paste0(dir_root,"MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.mimic+uw_n3000_epoch5_lr0.0002.nlg_eval_results.csv"))
enahnced4k_results = read.csv(paste0(dir_root,"MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.mimic+uw_n4000_epoch5_lr0.0002.nlg_eval_results.csv"))
enahnced5k_results = read.csv(paste0(dir_root,"MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.mimic+uw_n5000_epoch5_lr0.0002.nlg_eval_results.csv"))

all_results = cbind(baseline_results, enhanced2k_results[,2], enhanced3k_results[,2], enahnced4k_results[,2], enahnced5k_results[,2])
all_results1 = all_results[which(all_results$Metric %in% c("BERTC","BLEURTC","AggregateC","R1FC")),]
colnames(all_results1) = c("Metric","UW","UW+MIMIC2k","UW+MIMIC3k","UW+MIMIC4k","UW+MIMIC5k")

all_results1$Metric = c("ROUGE1","BERTScore","BLEU","Aggregate")
all_results2 = apply(all_results1[,-1],2,function(x){round(x,3)})
rownames(all_results2) = all_results1$Metric
# write out all_results2
write.csv(t(all_results2), "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/gemma/task3/all_results2.csv")

# print all_results2 in latex form
library(xtable)
print(xtable(t(all_results2), caption = "Performance comparison of different models on the validation set.", label = "tab:task3_results"), type = "latex")
# print out using latex and keep 3 digits
print(xtable(t(all_results2), caption = "Performance comparison of different models on the validation set.", label = "tab:task3_results"), type = "latex", digits = 3)
