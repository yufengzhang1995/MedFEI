library(ggplot2)
library(dplyr)
library(reshape2)

file_path = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step1.csv"
data = read.csv(file_path)

# read the phecode matching file
phecode_file = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/data/Phecode_map_v1_2_icd10cm_beta.csv"
phecode = read.csv(phecode_file)
colnames(phecode)[1] = c("ICD10")
phecode = phecode[,c("ICD10","phecode","exclude_name")]

head(phecode)

# merge the data column ICD10 with phecdoe column ICD10
data_temp = data
data_temp$phecode = phecode$phecode[match(data_temp$ICD10, phecode$ICD10)]
data_temp$exclude_name = phecode$exclude_name[match(data_temp$ICD10, phecode$ICD10)]

print(paste0("Proportion of ICD10 codes that have phecode mapping: ",mean(is.na(data_temp$phecode))))
# write.csv(data_temp, "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step2.csv")

## group by error
data2_temp = data_temp
data2_category = data2_temp[,c('id',"phecode","exclude_name","ICD10","Error.Flag")]
data2_category1 = data2_category%>%
    group_by(exclude_name,Error.Flag)%>%
    summarise(count = sum(!is.na(exclude_name))) %>%
    filter(exclude_name != "" & exclude_name != "NULL")

data2_category1$percentage = data2_category1$count/sum(data2_category1$count)
sum(data2_category1$percentage)
print(data2_category1[order(data2_category1$percentage, decreasing = TRUE),])
# write.out data2_category1
write.csv(data2_category1, "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/UW_trainData_diseaseDist.csv")

data2_category1$exclude_name = factor(data2_category1$exclude_name, levels = data2_category1$exclude_name)
data2_category1$Error.Flag = factor(data2_category1$Error.Flag, levels = unique(data2_category1$Error.Flag))
class(data2_category1$Error.Flag)
# plot the data, x=exclude_name, y=percentage, fill=Error.Flag, stack by Error.Flag
pic = ggplot(data2_category1, aes(x = exclude_name, y = percentage, fill = Error.Flag)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    xlab("") +
    ylab("Percentage") +
    ggtitle("Disease distribution")+
    theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 90, hjust = 1))+
    scale_fill_manual(values = c("red", "blue"))
# save the above pic
ggsave("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/UW_trainData_diseaseDist.png", plot = pic, width = 10, height = 6, units = "in", dpi = 300)
