UW_dist = read.csv("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step2_diseaseDist.csv")
MIMIC_dist = read.csv("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/MIMIC_diseaseDist.csv")

UW_dist$Dataset = "UW"
MIMIC_dist$Dataset = "MIMIC"
MIMIC_dist$Error.Flag = 2

apply(UW_dist,2,class)
apply(MIMIC_dist,2,class)

UW_dist$exclude_name = factor(UW_dist$exclude_name, levels = UW_dist$exclude_name)
UW_dist$Error.Flag = factor(UW_dist$Error.Flag, levels = unique(UW_dist$Error.Flag))
MIMIC_dist$exclude_name = factor(MIMIC_dist$exclude_name, levels = MIMIC_dist$exclude_name)
MIMIC_dist$Error.Flag = factor(MIMIC_dist$Error.Flag, levels = unique(MIMIC_dist$Error.Flag))
ggplot(UW_dist) +
    geom_bar(aes(x = exclude_name, y = percentage, fill = Error.Flag),stat = "identity",position = "stack") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    xlab("") +
    ylab("Percentage") +
    ggtitle("Disease distribution")+
    geom_bar(data = MIMIC_dist, aes(x = exclude_name, y = percentage,fill=Error.Flag), stat = "identity",position = "dodge")+
    theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 90, hjust = 1))
