# load the package for reading xlsx file
library(readxl)
library(dplyr)
library(reshape2)
library(ggplot2)
library(ggpubr)

######### Step 1: process the data #########

UW_results = read.csv('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/bert/metrics_UW.csv')
# combined_results = read.csv('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/bert/metrics_combined.csv')
combined2k_results = read.csv('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/bert/metrics_2k.csv')
combined3k_results = read.csv('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/bert/metrics_3k.csv')
combined4k_results = read.csv('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/bert/metrics_4k.csv')
combined5k_results = read.csv('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/bert/metrics_5k.csv')


result_proc = function(data){
    line_proc = function(x){
        temp = gsub(" ", "", x)
        temp = strsplit(as.character(temp), "|", fixed = TRUE)[[1]]
        return(temp)
    }
    result_new = data.frame(rbind(
        line_proc(data[3,1]),
        line_proc(data[5,1]),
        line_proc(data[7,1])))
    colnames(result_new) = c("","Stage", "Acc", "Spec","Pr", "Rc", "F1")
    return(result_new)
}

UW_results1 = result_proc(UW_results)
combined2k_results1 = result_proc(combined2k_results)
combined3k_results1 = result_proc(combined3k_results)
combined4k_results1 = result_proc(combined4k_results)
combined5k_results1 = result_proc(combined5k_results)

UW_results1$Dataset = "UW"
combined2k_results1$Dataset = "UW+MIMIC2k"
combined3k_results1$Dataset = "UW+MIMIC3k"
combined4k_results1$Dataset = "UW+MIMIC4k"
combined5k_results1$Dataset = "UW+MIMIC5k"

all_results = rbind(UW_results1, combined2k_results1, combined3k_results1, combined4k_results1, combined5k_results1)
all_results = all_results[,c(8,1:7)]

all_results$Stage = ifelse(all_results$Stage == "Firststage", "Task1", ifelse(all_results$Stage == "Secondstage-macro", "Task2", "Task2-micro"))
colnames(all_results)[3] = "Task"

all_results = all_results[,-2]
all_results

# write.csv(all_results, '/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/bert/all_results.csv', row.names = FALSE)

######### Step 2: plot the data #########
color_platte = c("Acc"="#f8996d", "Pr"="#6abe7a", "Rc" = "#639ba6", "F1"="#c189f2")

task1_results = all_results %>%
    filter(Task=="Task1") %>% 
    select("Dataset",names(color_platte))
task1_results

task1_results_long = melt(task1_results, id.vars = "Dataset")
colnames(task1_results_long) = c("Case", "Metric", "Value")
task1_results_long$Value = as.numeric(as.character(task1_results_long$Value))

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

# summarize task 2
task2_results = all_results %>%
    filter(Task=="Task2") %>% 
    select("Dataset",names(color_platte))

task2_results_long = melt(task2_results, id.vars = "Dataset")
colnames(task2_results_long) = c("Case", "Metric", "Value")
task2_results_long$Value = as.numeric(as.character(task2_results_long$Value))


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


(pic = ggarrange(pic1, pic2,  ncol = 2,
          labels = c("A", "B"),
          common.legend = TRUE, legend = "bottom"))
ggsave(filename = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/bert/bert_all_results.pdf", plot = pic, width = 6, height = 6, units = "in", dpi = 300)
