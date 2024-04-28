library(ggplot2)
library(dplyr)
library(reshape2)

# read data from /Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/MIMIC/merged_corrupted_mimic.csv
mimic_data = read.csv("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/MIMIC/merged_corrupted_mimic.csv")
colnames(mimic_data) # 
head(mimic_data)
mimic_data[493,]
dim(mimic_data)
dim(na.omit(mimic_data))

mimic_data2 = mimic_data[,c("X","index", "note_id", "subject_id", "primary_diagnosis", "icd10cm","phecode","icd10cm_str")]
# if the cell is "", let it be NA
mimic_data2[mimic_data2 == ""] = NA
na_index = which(is.na(mimic_data2$phecode))
mimic_data3 = mimic_data2[-na_index,]

# write the data to /Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/MIMIC/merged_corrupted_mimic.csv
write.csv(mimic_data3, "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/MIMIC/merged_corrupted_mimic2.csv", row.names = FALSE)

######### summarize the phecode distribution #########
rm(list=ls())
mimic_data_clean = read.csv("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/MIMIC/merged_corrupted_mimic2.csv")
# read the phecode matching file
phecode_file = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/data/Phecode_map_v1_2_icd10cm_beta.csv"
phecode = read.csv(phecode_file)
colnames(phecode)[1] = c("ICD10")
phecode = phecode[,c("phecode","exclude_name")]

head(phecode)
data_temp$exclude_name = phecode$exclude_name[match(data_temp$ICD10, phecode$ICD10)]

print(paste0("Proportion of ICD10 codes that have phecode mapping: ",mean(is.na(data_temp$phecode))))
# write.csv(data_temp, "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step2.csv")

mimic_data_clean$exclude_name = phecode$exclude_name[match(mimic_data_clean$phecode, phecode$phecode)]   
mimic_data_clean$id = 1:nrow(mimic_data_clean)   

data2_category = mimic_data_clean[,c('id',"icd10cm","phecode","exclude_name")]
data2_category1 = data2_category%>%
    group_by(exclude_name)%>%
    summarise(count = sum(!is.na(exclude_name))) %>%
    filter(exclude_name != "" & exclude_name != "NULL")

data2_category1$percentage = data2_category1$count/sum(data2_category1$count)
sum(data2_category1$percentage)
print(data2_category1[order(data2_category1$percentage, decreasing = TRUE),])
write.csv(data2_category1, "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/MIMIC_diseaseDist.csv")


ggplot(data2_category1, aes(x = exclude_name, y = percentage)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    xlab("") +
    ylab("Percentage") +
    ggtitle("Disease distribution")+
    theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 90, hjust = 1))
ggsave("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/MIMIC_diseaseDist.png", width = 10, height = 6, units = "in", dpi = 300)
