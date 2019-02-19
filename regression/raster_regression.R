setwd('D:/Thesis Practical/R/raster_mapping')

library(keras)
library(onehot)
library(sp)
library(raster)
library(rgdal)
library(rpart)
library(rasterVis)
library(caret)
library(snow)
library(rgeos)
library(tiff)
library(RStoolbox)
library(caTools)
source("Explorativeanalysis.R")
# remove objects from specified environment
rm(list=ls())
 
#read raster files
#fnf    <- brick('FNF.tif')
aspect.rast <- raster('Aspect.tif')
fnf    <- raster('FNF.tif')
AGB    = fnf

aspect <- brick('Aspect.tif')
slope  <- brick('Slope.tif')
palsar <- brick('palsar_final2.tif')
#s1vv   <- brick('S1_VV_final.tif')
s1vh   <- brick('S1_VH_final2.tif')
ndvi   <- brick('NDVI.tif')

#get NDVI proper band
ndvi_1band <- ndvi$NDVI.1

#get palsar proper bands from the stack
PalsarHH <- raster('palsar_final2.tif', band = 1)
PalsarHV <- raster('palsar_final2.tif', band = 2)
HH_savg  <- raster('palsar_final2.tif', band = 3)

#get Sentinel1 VH proper band
VH10_ent <- raster('S1_VH_final2.tif', band = 4)

#set the proper names for each band 
#colnames(palsar) <- c( 'PalsarHH', 'PalsarHV','HH_savg','HH_ent','HH_corr','HH_var','HV_savg','HV_ent','HV_corr','HV_var')
#colnames(s1vh)   <- c('Sent1VH_10','Sent1VH_12','VH10_ent','VH10_savg','VH10_corr','VH10_var','VH12_ent','VH12_savg','VH12_corr','VH12_var')

# combine aspect values into 8 categories 
to.categ <- function(x) {         #funtion to convert values inti categories
         ifelse(x > -1 & x <= 45, 1,
         ifelse(x > 45 & x <= 90, 2, 
         ifelse(x > 90 & x <= 135,3,
         ifelse(x > 135 & x <= 180,4,
         ifelse(x > 180 & x <= 225,5,
         ifelse(x > 225 & x <= 270,6,
         ifelse(x > 270 & x <= 315,7,8)))))))
}

aspect_categ <- calc(aspect, to.categ) #apply to.categ function on aspect


# apply one-hot encoding on aspect======================================================================
#data.new <- data.frame(na.omit(values(aspect_categ)))
aspect.df <- as.data.frame(aspect_categ)


#colnames(data.new) <- c('aspect_categ') 
#data.new$aspect_categ <- as.factor(data.new$aspect_categ)
#encoded <- onehot(data.new)
#data.new <- predict(encoded, data.new)
#aspect <- data.new[,-c(1,2,3,8)]# dropping not important ascpect categories
#aspect.df <- as.data.frame(aspect)


#=======================================================================================================
# convert the rasters into SpatialGridDataFrame

#aspect_categ  <- as(aspect_categ, "SpatialGridDataFrame")
#slope  <- as(slope, "SpatialGridDataFrame")
#palsar  <- as(palsar, "SpatialGridDataFrame")
#s1vv  <- as(s1vv, "SpatialGridDataFrame")
#s1vh  <- as(s1vh, "SpatialGridDataFrame")


#======================================================================================================
# stack the raster layers and confert into data frame
stacked <- stack(aspect_categ,slope, PalsarHH, PalsarHV, HH_savg, VH10_ent, ndvi_1band)
#data.new  <- as(stacked, "SpatialGridDataFrame")# doesn't work with one-hot encoding 

df.stacked <- as.data.frame(stacked)
#======================================================================================================
# apply one-hot encoding on aspect
df.stacked$layer <- as.factor(df.stacked$layer) 

encoded <- onehot(df.stacked)

df.stacked <- predict(encoded, df.stacked)

# dropping not important ascpect categories
v <- df.stacked[,-c(1,2,3,5,8)]#16,17

# rename the columns
colnames(v) <- c('Aspect_categ.4', 'Aspect_categ.6','Aspect_categ.7', 'Slope_Dem',
                 'PalsarHH', 'PalsarHV','HH_savg','VH10_ent','NDVI')
#v <- v[,c(1,2,3,4,5,9,10,12,7,8,6,11)]
v <- v[,c(1,2,3,4,5,6,11,7,8,9)]
#v.matrix <- as.matrix(v)

v <- as.data.frame(v)
#data scaling (min-max)
#scaled.data <- (v.matrix - min(v.matrix)) / (max(v.matrix)-min(v.matrix))

#================================= Training the regression model ===============================================

# read the data
data<- read.csv(file = "lriv_verjnakan.csv", header=T, sep = ",")# T is for true
#str(data) # inquire on the data structure

# calculate Root Mean Square Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}

#onehot encoding for categorical variable ============================================================

data$Aspect_categ <- as.factor(data$Aspect_categ)

encoded <- onehot(data)

data.new <- predict(encoded, data)

#data2 <- data[,c(-1,-3,-4)]
data2 <- data.new[,c(-1)]

#============================
# min-max Normalization
#============================
#scaled_data <- scale(data.new)
#scaled.data <- (data2 - min(data2)) / (max(data2)-min(data2))

x <- data.frame(data2)

selected <- x[,c(1,5,7,8,10,13,14,16,17,26)]# too good-> full
#selected <- x[,c(1,5,7,8,10,13,14,17,26,27,32)]
#selected <- x[,c(1,5,6,7,8, 10,11,12,14,16,17,19, 22)]
#selected <- x[,c(1,2,3,5,7,8,9,13)]

#=========================================#Modellling==================================================

#============================
require(MASS)
mod1lr = lm(AGB_pixel ~ .,data = selected)
summary(mod1lr)

#check this to adapt to the data
error <- mod1lr$residuals  # same as data$Y - predictedY
RMSE <- rmse(error)   # 0.004645207

# Steowise regression model
mod2lr = stepAIC(mod1lr, scope = list(upper = mod1lr$formula,lower = ~1), direction = "both")
summary(mod2lr)

error <- mod2lr$residuals  # same as data$Y - predictedY
RMSE <- rmse(error)

# selec the important variable
#selected <- x[,c(1,2,3,5,7,8,9, 13)]
# run the regression mdel with only selected variables
mod1lr = lm(AGB_pixel ~ .,data = selected)
summary(mod1lr)

plot(mod2lr, 1)

#==================================== AGB Mapping on the study area =============================================
agbY <- predict(mod1lr, v)
agbY.mtrx <- as.matrix(agbY)
AGB <- as.data.frame(agbY)

#convert data.frame back into raster filr ==============================================================
AGB_map = aspect.rast #copy the raster for the spatial information
# convert the dataframe into array with the dimensions of raster file
AGB.array = t(array(AGB$agbY, dim = c(aspect.rast@ncols,aspect.rast@nrows))) #aspect.df-> to be changed to predicted AGB

values(AGB_map) = AGB.array #get the values of AGB.array to the raster with the spatial info of the Aspect
plot(AGB_map)

#remove the values smaller than 0
remove.minuses <- function(x){x[x < 0] <- 0; return(x)}
remove.extras <- function(x){x[x > 45] <- 45; return(x)}

# remove negative values fro the raster
newAGB.map <- calc(AGB_map, remove.minuses)
newAGB.map <- calc(newAGB.map, remove.extras)
#convert back to data frame (the layer name will be by default= "layer)
AGB <- as.data.frame(newAGB.map)

# plot histogram
library(ggplot2)
library(reshape2)
qplot(AGB$layer, geom="histogram") 

qplot(AGB$layer,
      geom="histogram",
      binwidth = 3,  
      main = "Histogram of AGB volume", 
      xlab = "AGB T/ 0.09ha",  
      fill=I("brown"), 
      col=I("black"), 
      alpha=I(.2),
      xlim=c(0,42))

hist(newAGB.map,
     #breaks = 15,
     breaks = c(0, 5, 10, 15, 20, 25,30,42),
     main = "Aboveground forest biomass map",
     xlab = "Biomass (t/0,9ha)", ylab = "Frequency",
     col = "wheat3")

plot(newAGB.map)
#==================================write the AGB prediction map prediction map===========================
x <- writeRaster(newAGB.map, 'AGB_only_SAR.tif', overwrite=TRUE)