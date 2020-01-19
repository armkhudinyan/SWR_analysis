# setup the working directory
#setwd('D:/Thesis Practical/R/Scripts')
library(keras)
source("Explorativeanalysis.R")
library(onehot)
library(ggplot2)
library(GGally)
library(corrplot)
library(ggcorrplot)

# remove objects from specified environment
rm(list=ls())

rmse <- function(error)
{
  sqrt(mean(error^2))
}

# read the data
data<- read.csv(file = "lriv_verjnakan.csv", header=T, sep = ",")# T is for true

#str(data) # inquire on the data structure
#data2 <- data[,-c(1)]
#=============================================================================================
#DATA PREPARATION| correlation matrix

c<- data.frame(data)
M<-cor(c)
corrplot(M, method="square")
corr <- round(cor(c), 1)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "darkslateblue"), 
           title="Correlogram of input variables", 
           ggtheme=theme_bw)
#============================
#onehot encoding for categorical variable
#============================
data$Aspect_categ <- as.factor(data$Aspect_categ)

encoded <- onehot(data)

data.new <- predict(encoded, data)

#data2 <- data[,c(-1,-3,-4)]
data2 <- data.new[,-c(1)]


#============================
# min-max Normalization
#============================
#scaled_data <- scale(data.new)

#scaled_data$Aspect_categ <- scaled_data$encoded

#scaled.data <- (data2 - min(data2)) / (max(data2)-min(data2))
x <- data.frame(data2)

selected <- x[,c(1,11,12,15,16,17,18,19,20,21,22,23,24,25)]
selected <- x[,c(1,4,5,7,9,13,14,15,16,17,18)] 
selected <- x[,c(1,5,7,8,10,13,14,17,26)] # another good->SAR+ancill
selected <- x[,c(1,5,7,8,10,13,14,16,17,26)] # best selection-> full data combination
#y <- data[,c(2,3,4,5,6,7,8,9,10,11,12,13,14)]
#y <- x[,c(1,16,17,18,19,20,21,22,23)]
#ggpairs(y)
#selected <- x[,c(1,6,7,9,11,16,17,19)]
#write.csv(scaled_data, file='scaled2')

#============================
#Explorative analysis
#============================
#a = c('AGB_pixel','PalsarHV')
#pairs(x[,c(15,16)], lower.panel = panel.smooth, upper.panel = panel.cor)

#============================
#Modelling
#============================
require(MASS)
mod1lr = lm(AGB_pixel ~ .,data = selected)
summary(mod1lr)

#check this to adapt to the data
#error <- mod1lr$residuals  # same as data$Y - predictedY
#RMSE <- rmse(error)   # 0.004645207

mod2lr = stepAIC(mod1lr, scope = list(upper = mod1lr$formula,lower = ~1), direction = "both")
summary(mod2lr)

#model multicolinearity test
#car::vif(mod2lr)
Y <- mod2lr$fitted.values

#plot(data$AGB_pixel, Y, col = "black", pch=1)

error <- mod2lr$residuals  # same as data$Y - predictedY
RMSE <- rmse(error)
RMSE
error
plot(mod2lr, 1)
#===================================== 
#MODEL DIAGNOSTICS
#=====================================
#Evaluation asumptions about residuals
#normality distribution in residuasl
library(car)
library(ggpubr)

#qqPlot(mod2lr$residuals)
ggqqplot(mod2lr$residuals)

# shapiro-wilk normality test 
shapiro.test(mod2lr$residuals)
# From the output, the p-value > 0.05 implying that the distribution
# of the data are not significantly different from normal distribution. 
# In other words, we can assume the normality.
#=====================================
#another test is called kolmogorov-smirnov test...
#Homocedasticity (variance constant)
#=====================================
hist(mod2lr$residuals)

#Autocorrelation TEST CORRELOGRAM
#one-lag autoregressive approximation 
#to the true autocorrelation structure in ?? (error)
# here we regress the residual aginst itself to see the autocorelation
n = length(error) 
mod3 = lm(error[-n] ~ error[-1]) 
summary(mod3)
 

#===============================================
# Cross_Validation
#===============================================
library(caret)

#LOO cross-validation
# Define training control -> Leave One Out 
train.control <- trainControl(method = "LOOCV")
# Train the model
model <- train(AGB_pixel ~., data = selected, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

#===============================================
# repetitive K-fold cross-validation
# Define training control
set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 25, repeats = 3)
# Train the model
model <- train(AGB_pixel ~., data = x, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

#================================================
#  K-fold cross-validation
# Define training control
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model <- train(AGB_pixel ~., data = selected, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

#================================================
# Write CSV file
#================================================
out <- matrix(nrow = nrow(x), ncol = 4)
out[ , 1] <- data$plot_numb
out[ , 2] <- data$AGB_pixel
out[ , 3] <- error
out[ , 4] <- Y
out <- data.frame(out)
colnames(out) <- c('plot_nom', 'Reference_AGB','Prediction error', "Predicted_AGB")
#write.csv(out, "Model 1 predicted.csv")#olivedrab

# plot dependant variable against residuals
pda <- ggplot(out, aes(x = Reference_AGB, y = Predicted_AGB)) +
  geom_point(size = 2,color = "olivedrab", shape = 16) +
  geom_smooth(method="lm", se=T,size=1, color = "black") +
  geom_abline(intercept=0, slope=1, linetype=2, size=1, color = "red") +
  #theme_bw() +
  theme(text=element_text(size=11)) +
  labs(title = "Model 1",  
       theme(plot.title = element_text(color="black",hjust=0., size=12, face="bold")),
       subtitle = "R2 = 0.38    RMSE = 70 t/ha",  
       x = "Reference AGB (t/ha)", 
       y = "Predicted AGB (t/ha)") +
  theme(plot.subtitle=element_text(size=9, hjust=0,  color="black"))
plot(pda)

