# load the package for reading xlsx file
library(readxl)
library(dplyr)
library(reshape2)
library(ggplot2)

color_platte = c("Acc"="#f8996d", "Pr"="#6abe7a", "Rc" = "#639ba6", "F1"="#c189f2")

# read the baseline results
baseline <- read_excel('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/gemma/baseline.xlsx')
head(baseline)
colnames(baseline)
# extract results for task1
task1_baseline = baseline %>%
    select(1, grep('Task1', colnames(baseline)))
colnames(task1_baseline) <- gsub('Task1_', '', colnames(task1_baseline))
colnames(task1_baseline)[1] = "Case"
task1_baseline[1] = "UW"
task1_baseline = as.data.frame(task1_baseline)
task1_baseline

# extract results for task2
task2_baseline = baseline %>%
    select(1, grep('Task2', colnames(baseline)))
colnames(task2_baseline) <- gsub('Task2_', '', colnames(task2_baseline))
task2_baseline = task2_baseline[,!grepl('micro', colnames(task2_baseline))]
colnames(task2_baseline) = c("Case", "Acc", "Pr", "Rc", "F1")
task2_baseline$Case = "UW"
task2_baseline = as.data.frame(task2_baseline)

# read the xlsx file from '/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/gemma/results.xlsx'
gemma_results <- read_excel('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/gemma/Error_injection.xlsx')
head(gemma_results)
colnames(gemma_results)

# extract results for task1
task1_results = gemma_results %>%
    select(1, grep('Task1', colnames(gemma_results))) 
colnames(task1_results) <- gsub('Task1_', '', colnames(task1_results))
colnames(task1_results)[1] = "Case"
task1_results = task1_results[grepl('5 epoch', task1_results$Case),]
task1_results$Case = paste0('UW+MIMIC',c('2k','3k','4k','5k'), sep = "", "")
task1_results = rbind(task1_baseline, task1_results)

task1_results_long = melt(task1_results, id.vars = "Case")
colnames(task1_results_long) = c("Case", "Metric", "Value")
(pic1 = ggplot(task1_results_long, aes(x = Case, y = Value, group = Metric,color=Metric)) + 
    geom_point(size=3) +
    geom_line(size=1.5) + 
    xlab("Dataset")+
    ggtitle("Task 1") +
    scale_y_continuous(limits = c(0.4, 0.9)) +
    scale_color_manual(values = color_platte) +
    guides(size = FALSE) +
    theme_bw()+
    theme( axis.text.x = element_text(size = 12,angle = 60,hjust = 1,vjust = 1), 
            axis.text.y = element_text(size = 12), 
            legend.text = element_text(size = 12),
            legend.position = "bottom"))

# ggsave 
# ggsave(filename = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/gemma/task1_results.png", plot = pic1, width = 8, height = 4, units = "in", dpi = 300)

##### extract results for task2
task2_results = gemma_results %>%
    select(1, grep('Task2', colnames(gemma_results)))
colnames(task2_results) <- gsub('Task2_', '', colnames(task2_results))
task2_results
task2_results = task2_results[,!grepl('micro', colnames(task2_results))]
colnames(task2_results) = c("Case", "Acc", "Pr", "Rc", "F1")

task2_results = task2_results[grepl('5 epoch', task2_results$Case),]
task2_results$Case = paste0('UW+MIMIC',c('2k','3k','4k','5k'), sep = "", "")
task2_results = rbind(task2_baseline, task2_results)

task2_results_long = melt(task2_results, id.vars = "Case")
colnames(task2_results_long) = c("Case", "Metric", "Value")
(pic2 = ggplot(task2_results_long, aes(x = Case, y = Value, group = Metric,color=Metric)) + 
    geom_point(size=3) +
    geom_line(size=1.5) + 
    xlab("Dataset")+
    ylab("")+
    ggtitle("Task 2") +
    scale_y_continuous(limits = c(0.4, 0.9)) +
    scale_color_manual(values = color_platte) +
    guides(size = FALSE) +
    theme_bw()+
    theme( axis.text.x = element_text(size = 12,angle = 60,hjust = 1,vjust = 1), 
            axis.text.y = element_text(size = 12), 
            legend.text = element_text(size = 12),
            legend.position = "bottom"))

# # ggsave
# ggsave(filename = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/gemma/task2_results.png", plot = pic2, width = 8, height = 4, units = "in", dpi = 300)

# ggarrange task1 and task2
library(ggpubr)
(pic = ggarrange(pic1, pic2,  ncol = 2,
          labels = c("C", "D"),
          common.legend = TRUE, legend = "bottom"))
ggsave(filename = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/gemma/gemma_task1_task2_results.pdf", plot = pic, width = 6, height = 6, units = "in", dpi = 300)
