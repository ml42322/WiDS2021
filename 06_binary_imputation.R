df <- read.csv("training_clean.csv")
df_copy <- df
# install.packages("mice")
library("mice")
library("dplyr")

# Confirm unique values in column
unique(df$gcs_unable_apache)
# Convert to factor variable
df$gcs_unable_apache <- as.factor(df$gcs_unable_apache)
# Impute column
imputed_Data <- mice(df, m=5, maxit = 50, method = 'logreg', seed = 500)
# summary
summary(imputed_Data)
# check imputed values
imputed_Data$imp$gcs_unable_apache
# get new data
complete_data_1 <- complete(imputed_Data,1)
complete_data_2 <- complete(imputed_Data,2)
complete_data_3 <- complete(imputed_Data,3)
complete_data_4 <- complete(imputed_Data,4)
complete_data_5 <- complete(imputed_Data,5)
m1 <- as.data.frame(complete_data_1$gcs_unable_apache)
m2 <- as.data.frame(complete_data_2$gcs_unable_apache)
m3 <- as.data.frame(complete_data_3$gcs_unable_apache)
m4 <- as.data.frame(complete_data_4$gcs_unable_apache)
m5 <- as.data.frame(complete_data_5$gcs_unable_apache)

colnames(m1) <- "gcs_unable_apache"
colnames(m2) <- "gcs_unable_apache"
colnames(m3) <- "gcs_unable_apache"
colnames(m4) <- "gcs_unable_apache"
colnames(m5) <- "gcs_unable_apache"

dplyr::setdiff(m2,m1)
dplyr::setdiff(m2,m3)
dplyr::setdiff(m3,m4)
dplyr::setdiff(m4,m5)

# Looks like the fourth and fifth itterations are the same
# So let's just use the fifth iteration as her new complete column
m5 <- complete_data_5$gcs_unable_apache
df_copy$gcs_unable_apache <- m5

# Write the new csv
write.csv(df_copy, "training_clean_and_complete.csv")
