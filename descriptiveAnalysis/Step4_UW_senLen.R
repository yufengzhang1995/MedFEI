library(ggplot2)
library(dplyr)
library(reshape2)

file_path = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step1.csv"
data = read.csv(file_path)
x=data$Corrected.Sentence[2]
text = data$Text_index[2]

senLen = function(text){
    # if(x==""){
    #     return(c(NA))
    # }
    # split text by the symbol \n
    text_split = unlist(strsplit(text, "\n"))
    total_len = length(text_split)
    results = c(total_len)
    return(results)
}

data["Corrected.Sentence"]
# apply function find_index to columns "Corrected.Sentence" and "Text_index" in data
temp = apply(data, 1, function(x) senLen(x["Text_index"]))
data$total_len = temp

# density plot of total_len
(pic1 = ggplot(data, aes(x = total_len)) +
    geom_density(fill = "blue", color = "black") +
    xlab("Paragraph length by sentences") +
    ylab("Density") +
    theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 90, hjust = 1)) )
ggsave("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step4_senLen_densityplot.png", plot = pic1, width = 6, height = 6, units = "in", dpi = 300)

data2 = as.data.frame(table(data['total_len']))
colnames(data2) = c("Len", "Count")
data2$freq = data2$Count/sum(data2$Count)
data2$Len = as.numeric(as.character(data2$Len))

# bar plot of data_loc2,x=Location, y=freq
(pic2 = ggplot(data2, aes(x = Len, y = freq)) +
    geom_bar(stat = "identity", fill = "blue", color = "black") +
    xlab("Paragraph length by sentences") +
    ylab("Frequency") +
    scale_x_continuous(breaks = seq(0, 40, by = 5)) +
    theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 90, hjust = 1)))
# save the above pic
ggsave("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step4_senLen_barplot.png", plot = pic2, width = 6, height = 6, units = "in", dpi = 300)
