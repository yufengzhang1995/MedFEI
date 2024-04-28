library(ggplot2)
library(dplyr)
library(reshape2)

file_path = "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step1.csv"
data = read.csv(file_path)
x=data$Corrected.Sentence[2]
text = data$Text_index[2]

find_index = function(x,text){
    if(x==""){
        return(c(NA,NA))
    }
    # split text by the symbol \n
    text_split = unlist(strsplit(text, "\n"))
    # find the index of x in text_split
    grepl(x, text_split, fixed = TRUE)
    loc = grep(x, text_split, fixed = TRUE)
    total_len = length(text_split)
    results = c(loc, total_len)
    return(results)
}

# apply function find_index to columns "Corrected.Sentence" and "Text_index" in data
temp = apply(data, 1, function(x) find_index(x["Corrected.Sentence"],x["Text_index"]))
data$loc = unlist(lapply(temp, function(x) x[1]))
data$total_len = unlist(lapply(temp, function(x) x[2]))

data_loc = data$loc/data$total_len
data_loc = as.data.frame(data_loc[!is.na(data_loc)])
colnames(data_loc) = c("Location")
# density plot of data_loc
pic1 = ggplot(data_loc, aes(x = Location)) +
    geom_density(fill = "blue", color = "black") +
    xlab("Location") +
    ylab("Density") +
    ggtitle("Error location distribution")+
    theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 90, hjust = 1))
ggsave("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step3_diagnosisLoc_densityplot.png", plot = pic1, width = 6, height = 6, units = "in", dpi = 300)

data_loc2 = as.data.frame(table(data_loc))
colnames(data_loc2) = c("Location", "count")
data_loc2$freq = data_loc2$count/sum(data_loc2$count)
data_loc2$Location = as.numeric(as.character(data_loc2$Location))

# bar plot of data_loc2,x=Location, y=freq
pic2 = ggplot(data_loc2, aes(x = Location, y = freq)) +
    geom_bar(stat = "identity", fill = "blue", color = "black") +
    xlab("Location") +
    ylab("Frequency") +
    ggtitle("Error location distribution")+
    theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 90, hjust = 1))
# save the above pic
ggsave("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step3_diagnosisLoc_barplot.png", plot = pic2, width = 6, height = 6, units = "in", dpi = 300)
