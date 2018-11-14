# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# choose a work directory
mywd = "C:/Users/njmorris1/Downloads"
# mywd = "C:/Users/Nick Morris/Downloads"
setwd(mywd)

# create a name for a .txt file to log progress information while parallel processing
myfile = "log.txt"
file.create(myfile)

# cross validation folds
K = 2

# cross validation replications per fold
R = 5

# -----------------------------------------------------------------------------------
# ---- Packages ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# these are the packages i use

# data handling
require(data.table)
require(stringr)
require(tm)
require(stringdist)
require(gtools)
require(psych)

# plotting
require(VIM)
require(ggplot2)
require(gridExtra)
require(scales)
require(corrplot)
require(factoextra)

# modeling
require(forecast)
require(ranger)
require(e1071)
require(glmnet)
require(pROC)
require(caret)
require(cvTools)
require(SuperLearner)
require(xgboost)
require(h2o)
require(MLmetrics)

# parallel computing
require(foreach)
require(parallel)
require(doSNOW)
require(rlecuyer)

}

# -----------------------------------------------------------------------------------
# ---- Functions --------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- prints the data types of each column in a data frame -------------------------

types = function(dat)
{
  dat = data.frame(dat)
  
  column = sapply(1:ncol(dat), function(i) colnames(dat)[i])
  data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
  levels = sapply(1:ncol(dat), function(i) length(levels(dat[,i])))
  
  return(data.frame(column, data.type, levels))
}

# ---- a qualitative color scheme ---------------------------------------------------

mycolors = function(n)
{
  require(grDevices)
  return(colorRampPalette(c("#e41a1c", "#0099ff", "#4daf4a", "#984ea3", "#ff7f00", "#ff96ca", "#a65628"))(n))
}

# ---- generates a logarithmically spaced sequence ----------------------------------

lseq = function(from, to, length.out)
{
  return(exp(seq(log(from), log(to), length.out = length.out)))
}

# ---- builds a square confusion matrix ---------------------------------------------

confusion = function(ytrue, ypred)
{
  require(gtools)
  
  # make predicted and actual vectors into factors, if they aren't already
  if(class(ytrue) != "factor") ytrue = factor(ytrue)
  if(class(ypred) != "factor") ypred = factor(ypred)
  
  # combine their levels into one unique set of levels
  common.levels = mixedsort(unique(c(levels(ytrue), levels(ypred))))
  
  # give each vector the same levels
  ytrue = factor(ytrue, levels = common.levels)
  ypred = factor(ypred, levels = common.levels)
  
  # return a square confusion matrix
  return(table("Actual" = ytrue, "Predicted" = ypred))
}

# ---- runs goodness of fit tests across all columns of two data sets ---------------

sample.test = function(dat.sample, dat.remain, alpha = 0.5)
{
  # set up the types() function
  # this function extracts the column names, data types, and number of factor levels for each column of a data set
  types = function(dat)
  {
    dat = data.frame(dat)
    
    column = sapply(1:ncol(dat), function(i) colnames(dat)[i])
    data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
    levels = sapply(1:ncol(dat), function(i) length(levels(dat[,i])))
    
    return(data.frame(column, data.type, levels))
  }
  
  # make the data sets into data frames
  dat.sample = data.frame(dat.sample)
  dat.remain = data.frame(dat.remain)
  
  # get the data types of the data sets
  sample.types = types(dat.sample)
  remain.types = types(dat.remain)
  
  # ensure these data sets are identical
  if(identical(sample.types, remain.types))
  {
    # extract the column postion of factor variables
    factor.id = which(sample.types$data.type == "factor")
    
    # extract the column postion of numeric variables
    numeric.id = which(sample.types$data.type == "numeric" | sample.types$data.type == "integer")
    
    # get the p-values for the factor variables
    factor.test = lapply(factor.id, function(i)
    {
      # get the probability of each level of a factor occuring in dat.remain
      prob = as.numeric(table(dat.remain[,i]) / length(dat.remain[,i]))
      
      # get the frequency of each level of a factor occuring in dat.sample
      tab = table(dat.sample[,i])
      
      # perform a chi.sq test to reject or fail to reject the null hypothesis
      # the null: the observed frequency (tab) is equal to the expected count (prob)
      p.val = chisq.test(tab, p = prob)$p.value
      
      # determine if these variables are expected to come from the same distribution
      same.distribution = p.val > alpha
      
      # build a summary for variable i
      output = data.frame(variable = colnames(dat.sample)[i],
                          class = "factor",
                          gof.test = "chisq.test",
                          p.value = p.val,
                          alpha = alpha,
                          same.distribution = same.distribution)
      
      return(output)
    })
    
    # merge the list of rows into one table
    factor.test = do.call("rbind", factor.test)
    
    # get the p-values for the numeric variables
    numeric.test = lapply(numeric.id, function(i)
    {
      # perform a ks test to reject or fail to reject the null hypothesis
      # the null: the two variables come from the same distribution
      p.val = ks.test(dat.sample[,i], dat.remain[,i])$p.value
      
      # determine if these variables are expected to come from the same distribution
      same.distribution = p.val > alpha
      
      # build a summary for variable i
      output = data.frame(variable = colnames(dat.sample)[i],
                          class = "numeric",
                          gof.test = "ks.test",
                          p.value = p.val,
                          alpha = alpha,
                          same.distribution = same.distribution)
      
      return(output)
    })
    
    # merge the list of rows into one table
    numeric.test = do.call("rbind", numeric.test)
    
    # combine the test results into one table
    output = rbind(factor.test, numeric.test)
    
    return(output)
    
  } else
  {
    print("dat.sample and dat.remain must have the same:\n
          1. column names\n
          2. data class for each column\n
          3. number of levels for each factor column")
  }
}

# ---- creates an array for spliting up rows of a data set for cross validation -----

cv.folds = function(n, K, R, seed)
{
  # load required packages
  require(cvTools)
  require(data.table)
  
  # set the seed for repeatability
  set.seed(seed)
  
  # create the folds for repeated cross validation
  cv = cvFolds(n = n, K = K, R = R)
  
  # extract the fold id (which) and replication id (subsets)
  cv = data.table(cbind(cv$which, cv$subsets))
  
  # rename columns accordingly
  cv.names = c("fold", paste0("rep", seq(1:R)))
  setnames(cv, cv.names)
  
  # create the combinations of folds and replications
  # this is to make sure each fold is a test set once, per replication
  comb = expand.grid(fold = 1:K, rep = 1:R)
  
  # create a list, where each element is also a list where an element indicates which observations are in the training set and testing set for a model
  cv = lapply(1:nrow(comb), function(i)
  {
    # create the testing set
    testing = cv[fold == comb$fold[i]][[comb$rep[i] + 1]]
    
    # create the training set
    training = cv[fold != comb$fold[i]][[comb$rep[i] + 1]]
    
    # return the results in a list
    return(list(train = training, test = testing))
  })
  
  return(cv)
}

# ---- fast missing value imputation by chained random forests ----------------------

# got this from:
# https://github.com/mayer79/missRanger/blob/master/R/missRanger.R

missRanger <- function(data, maxiter = 10L, pmm.k = 0, seed = NULL, ...)
{
  cat("Missing value imputation by chained random forests")
  
  data = data.frame(data)
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  allVars <- names(which(sapply(data, function(z) (is.factor(z) || is.numeric(z)) && any(!is.na(z)))))
  
  if (length(allVars) < ncol(data)) {
    cat("\n  Variables ignored in imputation (wrong data type or all values missing: ")
    cat(setdiff(names(data), allVars), sep = ", ")
  }
  
  stopifnot(length(allVars) > 1L)
  data.na <- is.na(data[, allVars, drop = FALSE])
  count.seq <- sort(colMeans(data.na))
  visit.seq <- names(count.seq)[count.seq > 0]
  
  if (!length(visit.seq)) {
    return(data)
  }
  
  k <- 1L
  predError <- rep(1, length(visit.seq))
  names(predError) <- visit.seq
  crit <- TRUE
  completed <- setdiff(allVars, visit.seq)
  
  while (crit && k <= maxiter) {
    cat("\n  missRanger iteration ", k, ":", sep = "")
    data.last <- data
    predErrorLast <- predError
    
    for (v in visit.seq) {
      v.na <- data.na[, v]
      
      if (length(completed) == 0L) {
        data[, v] <- imputeUnivariate(data[, v])
      } else {
        fit <- ranger(formula = reformulate(completed, response = v), 
                      data = data[!v.na, union(v, completed)],
                      ...)
        pred <- predict(fit, data[v.na, allVars])$predictions
        data[v.na, v] <- if (pmm.k) pmm(fit$predictions, pred, data[!v.na, v], pmm.k) else pred
        predError[[v]] <- fit$prediction.error / (if (fit$treetype == "Regression") var(data[!v.na, v]) else 1)
        
        if (is.nan(predError[[v]])) {
          predError[[v]] <- 0
        }
      }
      
      completed <- union(completed, v)
      cat(".")
    }
    
    cat("done")
    k <- k + 1L
    crit <- mean(predError) < mean(predErrorLast)
  }
  
  cat("\n")
  if (k == 2L || (k == maxiter && crit)) data else data.last
}

}

# -----------------------------------------------------------------------------------
# ---- Prepare Data -----------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Checking Data Types ----------------------------------------------------------

{

# import the data
# for column descriptions see: https://www.kaggle.com/c/walmart-recruiting-trip-type-classification/data
train = data.table(read.csv("train.csv", na.strings = ""))
test = data.table(read.csv("test.csv", na.strings = ""))

# lets check out train
train
types(train)

# update columns that should be treated as a different data type
train[, TripType := factor(TripType, levels = sort(unique(TripType)))]
train[, Upc := factor(Upc, levels = sort(unique(Upc)))]
train[, FinelineNumber := factor(FinelineNumber, levels = sort(unique(FinelineNumber)))]

# lets check out test
test
types(test)

# update columns that should be treated as factors, not numbers
test[, Upc := factor(Upc, levels = sort(unique(Upc)))]
test[, FinelineNumber := factor(FinelineNumber, levels = sort(unique(FinelineNumber)))]

# give train and test an ID column
train[, ID := 1:nrow(train)]
test[, ID := 1:nrow(test) + nrow(train)]

# combine train and test to make sure all factors have the same levels
dat = data.table(rbind(train[,!"TripType"], test))

# split up train and test
id = max(train$ID)
train = data.table(cbind(TripType = train$TripType, dat[1:id]))
test = data.table(dat[(id + 1):nrow(dat)])

# remove objects we no longer need
rm(id, dat)

# free memory
gc()

}

# ---- Check for Missing Values -----------------------------------------------------

{

# this section is commented out because it takes a little bit to make the plots
# this was run once and the variables with missing values are mentioned at the bottom of this section

# lets check out if there are any missing values (NA's) in train
# aggr(train, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)

# lets check out if there are any missing values (NA's) in test
# aggr(test, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)

# Upc and FinelineNumber have missing values in both train and test

}

# ---- Imputations ------------------------------------------------------------------

{

# Upc and FinelineNumber also have are large amount of levels
# lets add the NA's as a level for these variables becuase imputations on factors with a large number of levels is impractical
train[is.na(Upc), Upc := "na"]
train[is.na(FinelineNumber), FinelineNumber := "na"]
test[is.na(Upc), Upc := "na"]
test[is.na(FinelineNumber), FinelineNumber := "na"]

}

}

# -----------------------------------------------------------------------------------
# ---- Feature Engineering ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# lets combine train and test
dat = data.table(rbind(train[,!"TripType"], test))

# ---- VisitNumber ------------------------------------------------------------------

{

# VisitNumber is the ID variable for the competition submission so lets first extract this from test before doing any feature engineering on VisitNumber
visit.num = test$VisitNumber

# lets also give visit.num a row ID
visit.num = data.table(id = 1:length(visit.num), value = visit.num)

# count how many times a number in VisitNumber was repeated to know the number of visits by a customer
visit.dat = table(dat$VisitNumber)

# make visit.dat into a table for joining purposes
visit.dat = data.table(VisitNumber = as.numeric(names(visit.dat)), Visits = as.numeric(visit.dat))

# set VisitNumber as the key column for joining purposes
setkey(visit.dat, VisitNumber)
setkey(dat, VisitNumber)

# join visit.dat onto dat
dat = visit.dat[dat]

# reorder dat by ID
dat = dat[order(ID)]

# remove objects we no longer need
rm(visit.dat)

# free memory
gc()

}

# ---- ScanCount  -------------------------------------------------------------------

{

# lets create variable called return, to indicate if a transaction was a return or a purchase
dat[, Return := as.numeric(ScanCount < 0)]

}

# ---- Department Description -------------------------------------------------------

{

# ---- frequent words ---------------------------------------------------------------

{

# lets text mine the Department Description column to find common words

# make all letters uppercase
Dept = toupper(as.character(dat$DepartmentDescription))

# remove all punctuation
Dept = removePunctuation(Dept)

# remove all numbers
Dept = removeNumbers(Dept)

# count the frequency of words
Dept.table = sort(table(Dept), decreasing = TRUE)

# make into a table
Dept.table = data.table(Dept.table)
setnames(Dept.table, c("word", "count"))

# make the column word into a factor for plotting purposes
Dept.table[, word := factor(word, levels = unique(word))]

# plot the proportion of presence for the top N words
N = 10

ggplot(Dept.table[1:N], aes(x = word, y = count / nrow(dat))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = percent) +
  labs(x = "Word", y = "Proportion of Presence") +
  theme_bw(15) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

# heres the proportion of Department Description covered by the top N words
sum(Dept.table$count[1:N] / nrow(dat))

# for each of the top N words create a binary variable for their presence
dept.vars = foreach(i = 1:N, .combine = "cbind") %do%
{
  # get the word
  var = Dept.table$word[i]
  
  # determine where it is present
  presence = as.numeric(grepl(paste0("\\<", var, "\\>"), Dept))
  
  return(presence)
}

# make dept.vars into a data.table and give it proper column names
dept.vars = data.table(dept.vars)
setnames(dept.vars, gsub(" ", ".", Dept.table$word[1:N]))

# combine dept.vars onto dat
dat = cbind(dat, dept.vars)

# remove objects we no longer need
rm(N, Dept, Dept.table, dept.vars, i, var, presence)

# free memory
gc()

}

# ---- simplier description ---------------------------------------------------------

{

# make all letters uppercase
Dept = toupper(as.character(dat$DepartmentDescription))

# remove all punctuation
Dept = removePunctuation(Dept)

# remove all numbers
Dept = removeNumbers(Dept)

# remove all spacing
Dept = gsub(" ", "", Dept)

# lets just use the first N characters of Department Description to reduce its levels
N = 1
Dept = substr(Dept, 1, N)
length(levels(factor(Dept)))

# add Dept to dat
dat[, Dept := factor(Dept)]

# remove DepartmentDescription as a variable becuase it has too many levels and has been replaced by other variables
dat[, DepartmentDescription := NULL]

# remove objects we no longer need
rm(N, Dept)

# free memory
gc()

}

}

# ---- FinelineNumber ---------------------------------------------------------------

{

# ---- similar size -----------------------------------------------------------------

{

# extract the numbers
Num = str_extract(dat$FinelineNumber, "[[:digit:]]+")

# count the number of digits
Num = nchar(Num)

# replace NA's in Num with 0's
Num[is.na(Num)] = 0

# give dat the variable we created
dat[, FNsize := Num]

# remove objects we no longer need
rm(Num)

# free memory
gc()

}

# ---- simplier code ----------------------------------------------------------------

{

# make FinelineNumber a character
Num = as.character(dat$FinelineNumber)

# lets just use the first N characters of FinelineNumber to reduce its levels
N = 1
Num = substr(Num, 1, N)
length(levels(factor(Num)))

# add Num to dat
dat[, FNcode := factor(Num)]

# remove FinelineNumber as a variable becuase it has too many levels and has been replaced by other variables
dat[, FinelineNumber := NULL]

# remove objects we no longer need
rm(N, Num)

# free memory
gc()

}

}

# ---- Upc --------------------------------------------------------------------------

{

# ---- similar size -----------------------------------------------------------------

{

# extract the numbers
UPC = str_extract(dat$Upc, "[[:digit:]]+")

# count the number of digits
UPC = nchar(UPC)

# replace NA's in UPC with 0's
UPC[is.na(UPC)] = 0

# give dat the variable we created
dat[, UPCsize := UPC]

# remove objects we no longer need
rm(UPC)

# free memory
gc()

}

# ---- simplier code ----------------------------------------------------------------

{

# make Upc a character
UPC = as.character(dat$Upc)

# lets just use the first N characters of Upc to reduce its levels
N = 1
UPC = substr(UPC, 1, N)
length(levels(factor(UPC)))

# add UPC to dat
dat[, UPCcode := factor(UPC)]

# remove Upc as a variable becuase it has too many levels and has been replaced by other variables
dat[, Upc := NULL]

# remove objects we no longer need
rm(UPC, N)

# free memory
gc()

}

}

# ---- TripType ---------------------------------------------------------------------

{

# TripType has too many levels so lets aggregate these levels together
# we will create a table to indicate what proportion of the new levels are made up of the old levels
# this table will allow us to map our predictions on the new levels back to the old levels

# plot a barplot of the level frequencies
tab = sort(table(train$TripType), decreasing = TRUE)
barplot(tab, las = 2)
tab

# level 39 (the second largest) seems like a good size to match up to
# this will be our cutoff value for combining levels
cutoff = as.numeric(tab[2])

# lets loop through tab, and give letters as new level IDs for each level in tab
# we will start at level 3 in tab because the first two levels are staying the same
start = 3

for(i in start:length(tab))
{
  # if this is the first iteration then initialize control variables
  if(i == start)
  {
    # new.level.size will keep track of the growth of a new level
    new.level.size = tab[i]
    
    # new.level.id will keep track of what letter of the alphabet (ie. first, second, etc) to assign as a new level
    new.level.id = 1
    
    # new.levels will become the vector indicating the assignment of old levels to new levels
    new.levels = LETTERS[new.level.id]
    
  # otherwise update control variables
  } else
  {
    # compute the size of the new level
    new.level.size = new.level.size + tab[i]
    
    # assign the this level in tab a new level according to the current value of new.level.id
    # and append this value to new.levels
    new.levels = c(new.levels, LETTERS[new.level.id])
    
    # if the size of the current new level violates the cutoff then:
      # reset new.level.size to 0
      # increment new.level.id by 1
    if(new.level.size >= cutoff)
    {
      new.level.size = 0
      new.level.id = new.level.id + 1
    }
  }
}

# give new.levels the first two levels in tab
new.levels = c(names(tab)[1:2], new.levels)

# create a table that maps old levels and new levels
map.levels = data.table(TripType = names(tab),
                        TripType.new = new.levels,
                        TripType.count = as.numeric(tab))

# set TripType as the key column in train and map.levels for joining purposes
setkey(train, TripType)
setkey(map.levels, TripType)

# lets join map.levels onto train
train = map.levels[train]

# remove TripType and TripType.count from train
train[, c("TripType", "TripType.count") := NULL]

# reorder train by ID
train = train[order(ID)]

# plot a barplot of the new level frequencies
tab = sort(table(train$TripType.new), decreasing = TRUE)
barplot(tab, las = 1)
tab

# make TripType.new into a factor
train[, TripType.new := factor(TripType.new, levels = unique(new.levels))]

# remove objects we no longer need
rm(cutoff, i, new.level.id, new.level.size, new.levels, tab, start)

# free memory
gc()

}

# ---- Representative Sampling ------------------------------------------------------

{

# there are a lot of rows in train and this will slow down model build ing significantly
# so lets determine which rows to randomly sample from train
train = cbind(TripType.new = train$TripType.new, dat[ID %in% train$ID])

# ---- Iteration 1 ------------------------------------------------------------------

# lets take a random stratified sample of train based on TripType.new
# compute the proportion of each level of TripType.new
prop = table(train$TripType.new) / nrow(train)

# choose a reasonable number of rows to work with
N = 5000

# compute how many random samples we need of each level of TripType.new to have a representative population of size N (approx.)
pop = round(ceiling(N * prop), 0)

# randomly sample IDs of train according to pop
set.seed(42)
IDs = sort(unlist(lapply(1:length(pop), function(i) sample(train[TripType.new == names(pop)[i], ID], pop[i]))))

# lets use IDs to create our sample of train
train.sample = data.table(train[ID %in% IDs])

# extract the portion of train that we didn't sample
train.remain = data.table(train[!(ID %in% IDs)])

# lets run goodness of fit tests on train.sample to ensure that it is a representative sample of train
train.test = data.table(sample.test(dat.sample = train.sample[,!"ID"], 
                                    dat.remain = train.remain[,!"ID"],
                                    alpha = 0.5))

# alpha is the level of significance, and it can take a value between, but not including, 0 and 1
# so in gof testing, we would hypothesize that we are (1 - alpha)% confident that the same variable from train.sample and train.remain DONT come from the same probability distribution
# therefore we want p.values no smaller than a high alpha to have little confidence about this hypothesis, and accept that the same variable from train.sample and train.remain DO come from the same probability distribution
# find out if any p.values violated alpha
train.test[same.distribution == "FALSE"]

# ---- Iteration 2 ------------------------------------------------------------------

# VisitNumber has the worst violation so lets add this in our sampling
# let make a categorical version of VisitNumber using bins of a histogram to make stratified random sampling easier
train[, strata := cut(VisitNumber, hist(train$VisitNumber, breaks = 4)$breaks)]

# add TripType.new to strata and make strata a factor variable
train[, strata := factor(paste(strata, TripType.new, sep = "."))]

# lets compute the proportion of each level in strata
prop = table(train$strata) / nrow(train)

# lets compute how many random samples we need of each level in strata to have a representative population of size N (approx.)
pop = round(ceiling(N * prop), 0)

# lets randomly sample train according to pop
set.seed(42)
IDs = sort(unlist(lapply(1:length(pop), function(i) sample(train[strata == names(pop)[i], ID], pop[i]))))

# lets use IDs to create our sample of train
train.sample = data.table(train[ID %in% IDs])

# extract the portion of train that we didn't sample
train.remain = data.table(train[!(ID %in% IDs)])

# run the gof tests
train.test = data.table(sample.test(dat.sample = train.sample[,!c("ID", "strata")], 
                                    dat.remain = train.remain[,!c("ID", "strata")],
                                    alpha = 0.5))

# find out if any p.values violated alpha
train.test[same.distribution == "FALSE"]

# remove objects we no longer need
rm(prop, N, pop, train.remain, train.test)

# free memory
gc()

}

# ---- Scaling ----------------------------------------------------------------------

{

# remove the ID column so we can scale the data
dat[, ID := NULL]

# reformat all columns to be numeric by creating dummy variables for factor columns
dat = data.table(model.matrix(~., dat)[,-1])

# scale dat so that all variables can be compared fairly
dat = data.table(scale(dat))

# give the ID column back to dat
dat[, ID := 1:nrow(dat)]

# split up dat into train and test
train = cbind(TripType.new = train$TripType.new, dat[ID %in% train$ID])
test = dat[ID %in% test$ID]

# remove the ID column from train and test
train[, ID := NULL]
test[, ID := NULL]

# remove objects we no longer need
rm(dat)

# free memory
gc()

}

# extract the representative sample from train for modeling
train.sample = data.table(train[IDs])

}

# -----------------------------------------------------------------------------------
# ---- Feature Selection ------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- ANOVA ------------------------------------------------------------------------

{

# build a copy of train for aov
aov.dat = data.table(train.sample[,!"TripType.new"])

# build all two-way interactions
aov.dat = data.table(model.matrix(~.^2, aov.dat)[,-1])

# create column names for aov.dat
# remove all `
aov.names = gsub("`", "", names(aov.dat))

# remove all spacing
aov.names = gsub(" ", "", aov.names)

# replace ":" with "."
aov.names = gsub(":", ".", aov.names)

# set the names of aov.dat
setnames(aov.dat, aov.names)

# make the response variable numeric
res.dat = data.table(model.matrix(~., train.sample[,.(TripType.new)])[,-1])

# create column names for res.dat
# remove all `
res.names = gsub("`", "", names(res.dat))

# remove all spacing
res.names = gsub(" ", "", res.names)

# set the names of res.dat
setnames(res.dat, res.names)

# create a formula for anova
lhs = paste(res.names, collapse = " + ")
form = as.formula(paste(lhs, "~."))

# attach res.dat to aov.dat
aov.dat = cbind(res.dat, aov.dat)

# free memory
gc()

# ---- Cut 1: Keep variables with p-value < 0.50 -----------------------------------

# build an anova table
my.aov = aov(formula = form, data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.5
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.5, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# setup aov.dat to have the variables in keep.var
aov.dat = data.table(aov.dat[, c(res.names, keep.var), with = FALSE])

# ---- Cut 2: Keep variables with p-value < 0.25 -----------------------------------

# build an anova table
my.aov = aov(formula = form, data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.5
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.25, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# setup aov.dat to have the variables in keep.var
aov.dat = data.table(aov.dat[, c(res.names, keep.var), with = FALSE])

# ---- Cut 3: Keep variables with p-value < 0.10 -----------------------------------

# build an anova table
my.aov = aov(formula = form, data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.1
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.1, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# make sure only aov.names are in keep.var
keep.var = keep.var[which((keep.var %in% aov.names) == TRUE)]

# remove objects we no longer need
rm(aov.dat, my.aov, aov.names, res.dat, res.names, form, lhs)

# free memory
gc()

}

# ---- Correlation ------------------------------------------------------------------

{

# build a copy of train.sample for modeling
mod.dat = data.table(train.sample)

# extract all potential variables
cor.dat = data.table(mod.dat[,!"TripType.new"])

# reformat all columns to be numeric by creating dummy variables for factor columns
cor.dat = data.table(model.matrix(~.^2, cor.dat)[,-1])

# create column names for cor.dat
# remove all `
cor.names = gsub("`", "", names(cor.dat))

# remove all spacing
cor.names = gsub(" ", "", cor.names)

# replace ":" with "."
cor.names = gsub(":", ".", cor.names)

# set the names of cor.dat
setnames(cor.dat, cor.names)

# attach cor.dat to mod.dat
mod.dat = cbind(TripType.new = mod.dat$TripType.new, cor.dat)

# setup mod.dat and cor.dat to have the variables in keep.var
mod.dat = data.table(mod.dat[, c("TripType.new", keep.var), with = FALSE])
cor.dat = data.table(cor.dat[, keep.var, with = FALSE])

# compute correlations
cors = cor(cor.dat)
# replace any NA's with 1's
cors[is.na(cors)] = 1

# find out which variables are highly correlated (>= 0.9) and remove them
find.dat = findCorrelation(cors, cutoff = 0.9, names = TRUE)

# remove columns from mod.dat according to find.dat
if(length(find.dat) > 0) mod.dat = mod.dat[, !find.dat, with = FALSE]

}

# ---- Importance -------------------------------------------------------------------

{

# the classes are imbalanced so lets define the case.weights parameter where a class with more observations is case.weightsed less
case.weights = table(train$TripType.new)
case.weights = max(case.weights) / case.weights

# make case.weights into a table so we can join it onto mod.dat
case.weights = data.table(TripType.new = names(case.weights), 
                          value = as.numeric(case.weights))

# give mod.dat an ID column so we can maintain the original order of rows
mod.dat[, ID := 1:nrow(mod.dat)]

# set TripType.new as the key column for joining purposes
setkey(mod.dat, TripType.new)
setkey(case.weights, TripType.new)

# join case.weights onto mod.dat
case.weights = data.table(case.weights[mod.dat])

# order case.weights by ID and extract the value column of case.weights
case.weights = case.weights[order(ID)]
case.weights = case.weights$value

# remove the ID column in mod.dat
mod.dat[, ID := NULL]

# choose how many tasks and threads to use
tasks = 10
num.threads = 16

# set up seeds for reproducability
set.seed(42)
seeds = sample(1:1000, tasks)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("random forest - variable importance\n")
cat(paste("task 1 started at", Sys.time()), "\n")
sink()

# build random forest models
var.imp = foreach(i = 1:tasks) %do%
{
  # build the random forest
  mod = ranger(TripType.new ~ ., 
               data = mod.dat,
               num.trees = 1000,
               case.weights = case.weights,
               num.threads = num.threads,
               seed = seeds[i],
               importance = "impurity")
  
  # extract variable importance
  imp = importance(mod)
  
  # make imp into a table
  imp = data.table(variable = names(imp), 
                   value = as.numeric(imp))
  
  # order by variable name
  imp = imp[mixedorder(variable)]
  
  # add the task number to imp
  imp[, task := i]
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time()), "\n")
  sink()
  
  return(imp)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time()), "\n")
sink()

# combine the list of data tables into one table
var.imp = rbindlist(var.imp)

# put importance on a 0-1 scale for easy comparison
var.imp[, value := rescale(value)]

# average importance of variables
var.imp = var.imp[, .(value = mean(value)), by = .(variable)]

# order by importance
var.imp = var.imp[order(value, decreasing = TRUE)]

# make variable a factor for plotting purposes
var.imp[, variable := factor(variable, levels = unique(variable))]

# plot a barplot of variable importance
ggplot(var.imp, aes(x = variable, y = value, fill = value, color = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Variable", y = "Importance") +
  scale_y_continuous(labels = percent) +
  scale_fill_gradient(low = "yellow", high = "red") +
  scale_color_gradient(low = "yellow", high = "red") +
  theme_dark(15) +
  theme(legend.position = "none", axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major.x = element_blank())

# lets only keep variables with at least 20% importance
keep.dat = gsub("`", "", var.imp[value >= 0.20, variable])
mod.dat = mod.dat[, c("TripType.new", keep.dat), with = FALSE]

# heres our variables for TripType.new
TripType.new.variables = keep.dat

}

# ---- Finalize Data ----------------------------------------------------------------

{

# setup train to have the variables in TripType.new.variables
# lets adjust the terms in TripType.new.variables such that they can be written in a formula arguement
# we need to specify them in a formula to build just the terms of interest
# if we were to just build all possible two way interactions then R would crash becuase it would be too large of a task for a computer with 32 GB of RAM
# so we will build just the terms we need according to TripType.new.variables

# replace '.' with ':'
terms = gsub("\\.", ":", TripType.new.variables)

# extract variables in train that should have periods
periods = names(train)[grepl("\\.", names(train))]

# if the variables in 'periods' are also somewhere in 'terms' then substitute them in
for(i in periods)
{
  terms = gsub(gsub("\\.", ":", i), i, terms)
}

# lets make terms into a formula
terms = as.formula(paste("~", paste(terms, collapse = " + ")))

# use 'terms' to build the variables in 'TripType.new.variables' for train
train = cbind(TripType.new = train$TripType.new, 
              data.table(model.matrix(terms, train[, !"TripType.new"])[,-1]))

# lets give train a better set of column names
train.names = names(train)

# replace ":" with "."
train.names = gsub(":", ".", train.names)

# rename the columns of train
setnames(train, train.names)

# setup test to have the variables in TripType.new.variables
# use 'terms' to build the variables in 'TripType.new.variables' for test
test = data.table(model.matrix(terms, test)[,-1])

# lets give test a better set of column names
test.names = names(test)

# replace ":" with "."
test.names = gsub(":", ".", test.names)

# rename the columns of test
setnames(test, test.names)

# extract the representative sample from train for modeling
train.sample = data.table(train[IDs])

# remove objects we no longer need
rm(seeds, tasks, test.names, cor.names, keep.var, num.threads,
   TripType.new.variables, cor.dat, cors, find.dat, var.imp, mod.dat, 
   keep.dat, i, periods, terms, train.names, imp, mod, case.weights)

# free memory
gc()

}

}

# -----------------------------------------------------------------------------------
# ---- Gradient Boosting Model ------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# we have 7 hyperparameters of interest:
  # nrounds ~ the max number of boosting iterations
  # eta ~ the learning rate
  # max_depth ~ maximum depth of a tree
  # min_child_weight ~ minimum sum of instance weight needed in a child
  # gamma ~ minimum loss reduction required to make a further partition on a leaf node of the tree
  # subsample ~ the proportion of data (rows) to randomly sample each round
  # colsample_bytree ~ the proportion of variables (columns) to randomly sample each round

# check out this link for help on tuning:
  # https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur
  # google stuff and you'll find other approaches

# extract predictors (X) and response (Y)
X = as.matrix(train.sample[,!"TripType.new"])
Y = as.numeric(train.sample$TripType.new) - 1
Y.max = max(Y)

# create parameter combinations to test
doe = data.table(expand.grid(nrounds = 100,
                             eta = 0.1,
                             max_depth = c(4, 6, 8, 10, 12), 
                             min_child_weight = c(1, 3, 5, 7, 9),
                             gamma = 0,
                             subsample = 1,
                             colsample_bytree = 1))

# build the cross validation folds
cv = cv.folds(n = nrow(X), K = K, R = R, seed = 42)

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# the classes are imbalanced so lets define the weight parameter where a class with more observations is weighted less
weight = table(train.sample$TripType.new)
weight = max(weight) / weight

# make weight into a table so we can join it onto train.sample
weight = data.table(TripType.new = names(weight), 
                    value = as.numeric(weight))

# give train.sample an ID column so we can maintain the original order of rows
train.sample[, ID := 1:nrow(train.sample)]

# set TripType.new as the key column for joining purposes
setkey(train.sample, TripType.new)
setkey(weight, TripType.new)

# join weight onto train.sample
weight = data.table(weight[train.sample])

# order weight by ID and extract the value column of weight
weight = weight[order(ID)]
weight = weight$value

# remove the ID column in train.sample
train.sample[, ID := NULL]

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
gbm.pred = function(Xtrain, Ytrain, Xtest, Ytest, objective, eval_metric, eta, max_depth, nrounds, min_child_weight, gamma, subsample, colsample_bytree, num_class, weight, nthread)
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Ytrain, data = Xtrain,
                objective = objective, eval_metric = eval_metric,
                eta = eta, max_depth = max_depth, num_class = num_class,
                nrounds = nrounds, min_child_weight = min_child_weight,
                gamma = gamma, verbose = 0, weight = weight, nthread = nthread,
                subsample = subsample, colsample_bytree = colsample_bytree)
  
  # ---- multi log loss metric ----
  
  # extract the probability prediction matrix
  ypred = predict(mod, newdata = Xtest, reshape = TRUE)
  
  # build a matrix indicating the true class values
  ytrue = model.matrix(~., data = data.frame(factor(Ytest, levels = 0:Y.max)))[,-1]
  ytrue = cbind(1 - rowSums(ytrue), ytrue)
  
  # compute the multi-class logarithmic loss
  mll = MultiLogLoss(y_pred = ypred, y_true = ytrue)
  
  # ---- Kappa ----
  
  # extract the predicted classes and actual classes
  ypred = apply(ypred, 1, which.max) - 1
  ytrue = factor(Ytest, levels = 0:Y.max)
  
  # build a square confusion matrix
  conf = confusion(ytrue = ytrue, ypred = ypred)
  
  # get the total number of observations
  n = sum(conf)
  
  # get the vector of correct predictions
  dia = diag(conf)
  
  # get the vector of the number of observations per class
  rsum = rowSums(conf)
  
  # get the vector of the number of predictions per class
  csum = colSums(conf)
  
  # get the proportion of observations per class
  p = rsum / n
  
  # get the proportion of predcitions per class
  q = csum / n
  
  # compute accuracy
  acc = sum(dia) / n
  
  # compute expected accuracy
  exp.acc = sum(p * q)
  
  # compute kappa
  kap = (acc - exp.acc) / (1 - exp.acc)
  
  # ---- one-vs-all metrics ----
  
  # compute a binary confusion matrix for each class
  one.v.all = lapply(1:nrow(conf), function(i)
  {
    # extract the four entries of a binary confusion matrix
    v = c(conf[i,i], 
          rsum[i] - conf[i,i], 
          csum[i] - conf[i,i], 
          n - rsum[i] - csum[i] + conf[i,i]);
    
    # build the confusion matrix
    return(matrix(v, nrow = 2, byrow = TRUE))
  })
  
  # sum up all of the matrices
  one.v.all = Reduce('+', one.v.all)
  
  # compute the micro average accuracy
  micro.acc = sum(diag(one.v.all)) / sum(one.v.all)
  
  # get the macro accuracy
  macro.acc = acc
  
  # combine all of our performance metrics
  output = data.table(Multi.Log.Loss = mll, Kappa = kap,
                      Macro.Accuracy = macro.acc, Micro.Accuracy = micro.acc)
  
  return(output)
}

# choose the number of workers/threads and tasks for parallel processing
# specifying a value > 1 for workers means that multiple models in doe will be built in parallel
# specifying a value > 1 for nthread means that each model will internally be built in parallel
workers = 16
nthread = 1
tasks = nrow(doe)

# set up a cluster if workers > 1, otherwise don't set up a cluster
if(workers > 1)
{
  # setup parallel processing
  cl = makeCluster(workers, type = "SOCK", outfile = "")
  registerDoSNOW(cl)
  
  # define %dopar%
  `%fun%` = `%dopar%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("gradient boosting - cross validation\n")
  cat(paste(workers, "workers started at", Sys.time(), "\n"))
  sink()
  
} else
{
  # define %do%
  `%fun%` = `%do%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("gradient boosting - cross validation\n")
  cat(paste("task 1 started at", Sys.time(), "\n"))
  sink()
}

# perform cross validation for each of the models in doe
gbm.cv = foreach(i = 1:tasks) %fun%
{
  # load packages we need for our tasks
  require(data.table)
  require(xgboost)
  require(MLmetrics)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # extract the training set subsection of weight
  weight.train = weight[folds$train]
  
  # build model and get prediction results
  output = gbm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    objective = "multi:softprob", eval_metric = "merror", num_class = Y.max + 1,
                    eta = doe$eta[i], max_depth = doe$max_depth[i], nrounds = doe$nrounds[i], 
                    min_child_weight = doe$min_child_weight[i], gamma = doe$gamma[i], 
                    subsample = doe$subsample[i], colsample_bytree = doe$colsample_bytree[i],
                    weight = weight.train, nthread = nthread)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end the cluster if it was set up
if(workers > 1)
{
  stopCluster(cl)
}

# combine the list of tables into one table
gbm.cv = rbindlist(gbm.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

gbm.diag = gbm.cv[,.(stat = factor(stat, levels = stat),
                     Multi.Log.Loss = as.vector(summary(na.omit(Multi.Log.Loss))),
                     Kappa = as.vector(summary(na.omit(Kappa))),
                     Macro.Accuracy = as.vector(summary(na.omit(Macro.Accuracy))),
                     Micro.Accuracy = as.vector(summary(na.omit(Micro.Accuracy)))),
                  by = .(eta, max_depth, nrounds, min_child_weight, gamma, subsample, colsample_bytree)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(gbm.diag)
gbm.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert gbm.diag into long format for plotting purposes
DT = data.table(melt(gbm.diag, measure.vars = c("Multi.Log.Loss", "Kappa", "Macro.Accuracy", "Micro.Accuracy")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
gbm.diag[stat == "Mean" & Kappa >= 0.35 & mod %in% gbm.diag[stat == "Median" & Multi.Log.Loss <= 1.35, mod]]

# model 16 looks good
gbm.diag = gbm.diag[mod == 16]

# rename model to gbm
gbm.diag[, mod := rep("gbm", nrow(gbm.diag))]

# recall num_class
Y.max + 1

# build our model
set.seed(42)
gbm.mod = xgboost(label = Y, data = X, objective = "multi:softprob", eval_metric = "merror", 
                  num_class = 6, eta = 0.1, max_depth = 4, nrounds = 100, min_child_weight = 7, 
                  gamma = 0, subsample = 1, colsample_bytree = 1, weight = weight, verbose = 0)

# store model diagnostic results
gbm.diag = gbm.diag[,.(Multi.Log.Loss, Kappa, Macro.Accuracy, Micro.Accuracy, stat, mod)]
mods.diag = data.table(gbm.diag)

# store the model
gbm.list = list("mod" = gbm.mod)
mods.list = list()
mods.list$gbm = gbm.list

# remove objects we no longer need
if(!(workers > 1))
{
  rm(output, Xtest, Xtrain, i, Ytest, Ytrain, weight.train, folds)
}

rm(gbm.cv, gbm.diag, gbm.list, gbm.mod, gbm.pred, doe, DT, weight, nthread,
   X, Y, diag.plot, cl, workers, tasks, num.stats, num.rows, stat, Y.max, `%fun%`)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Random Forest Model ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# we have 1 hyperparameter of interest:
  # min.node.size ~ minimum size of terminal nodes (ie. the minimum number of data points that can be grouped together in any node of a tree)

# check out this link for help on tuning:
  # https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur

# extract predictors (X) and response (Y)
X = data.table(train.sample[,!"TripType.new"])
Y = as.factor(as.numeric(train.sample$TripType.new) - 1)
Y.max = max(as.numeric(train.sample$TripType.new) - 1)

# create parameter combinations to test
doe = data.table(expand.grid(min.node.size = seq(1, 21, 2)))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# the classes are imbalanced so lets define the case.weights parameter where a class with more observations is weighted less
case.weights = table(train.sample$TripType.new)
case.weights = max(case.weights) / case.weights

# make case.weights into a table so we can join it onto train.sample
case.weights = data.table(TripType.new = names(case.weights), 
                    value = as.numeric(case.weights))

# give train.sample an ID column so we can maintain the original order of rows
train.sample[, ID := 1:nrow(train.sample)]

# set TripType.new as the key column for joining purposes
setkey(train.sample, TripType.new)
setkey(case.weights, TripType.new)

# join case.weights onto train.sample
case.weights = data.table(case.weights[train.sample])

# order case.weights by ID and extract the value column of case.weights
case.weights = case.weights[order(ID)]
case.weights = case.weights$value

# remove the ID column in train.sample
train.sample[, ID := NULL]

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
rf.pred = function(Xtrain, Ytrain, Xtest, Ytest, min.node.size, case.weights, num.threads)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # build the training model
  mod = ranger(y ~ ., 
               data = dat,
               min.node.size = min.node.size,
               case.weights = case.weights,
               num.threads = num.threads,
               probability = TRUE,
               seed = 42)
  
  # ---- multi log loss metric ----
  
  # extract the probability prediction matrix
  ypred = predict(mod, data = Xtest, num.threads = num.threads)$predictions
  
  # build a matrix indicating the true class values
  ytrue = model.matrix(~., data = data.frame(factor(Ytest, levels = 0:Y.max)))[,-1]
  ytrue = cbind(1 - rowSums(ytrue), ytrue)
  
  # compute the multi-class logarithmic loss
  mll = MultiLogLoss(y_pred = ypred, y_true = ytrue)
  
  # ---- Kappa ----
  
  # extract the predicted classes and actual classes
  ypred = apply(ypred, 1, which.max) - 1
  ytrue = factor(Ytest, levels = 0:Y.max)
  
  # build a square confusion matrix
  conf = confusion(ytrue = ytrue, ypred = ypred)
  
  # get the total number of observations
  n = sum(conf)
  
  # get the vector of correct predictions
  dia = diag(conf)
  
  # get the vector of the number of observations per class
  rsum = rowSums(conf)
  
  # get the vector of the number of predictions per class
  csum = colSums(conf)
  
  # get the proportion of observations per class
  p = rsum / n
  
  # get the proportion of predcitions per class
  q = csum / n
  
  # compute accuracy
  acc = sum(dia) / n
  
  # compute expected accuracy
  exp.acc = sum(p * q)
  
  # compute kappa
  kap = (acc - exp.acc) / (1 - exp.acc)
  
  # ---- one-vs-all metrics ----
  
  # compute a binary confusion matrix for each class
  one.v.all = lapply(1:nrow(conf), function(i)
  {
    # extract the four entries of a binary confusion matrix
    v = c(conf[i,i], 
          rsum[i] - conf[i,i], 
          csum[i] - conf[i,i], 
          n - rsum[i] - csum[i] + conf[i,i]);
    
    # build the confusion matrix
    return(matrix(v, nrow = 2, byrow = TRUE))
  })
  
  # sum up all of the matrices
  one.v.all = Reduce('+', one.v.all)
  
  # compute the micro average accuracy
  micro.acc = sum(diag(one.v.all)) / sum(one.v.all)
  
  # get the macro accuracy
  macro.acc = acc
  
  # combine all of our performance metrics
  output = data.table(Multi.Log.Loss = mll, Kappa = kap,
                    Macro.Accuracy = macro.acc, Micro.Accuracy = micro.acc)
  
  return(output)
}

# choose the number of workers/threads and tasks for parallel processing
# specifying a value > 1 for workers means that multiple models in doe will be built in parallel
# specifying a value > 1 for num.threads means that each model will internally be built in parallel
workers = 16
num.threads = 1
tasks = nrow(doe)

# set up a cluster if workers > 1, otherwise don't set up a cluster
if(workers > 1)
{
  # setup parallel processing
  cl = makeCluster(workers, type = "SOCK", outfile = "")
  registerDoSNOW(cl)
  
  # define %dopar%
  `%fun%` = `%dopar%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("random forest - cross validation\n")
  cat(paste(workers, "workers started at", Sys.time(), "\n"))
  sink()
  
} else
{
  # define %do%
  `%fun%` = `%do%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("random forest - cross validation\n")
  cat(paste("task 1 started at", Sys.time(), "\n"))
  sink()
}

# perform cross validation for each of the models in doe
rf.cv = foreach(i = 1:tasks) %fun%
{
  # load packages we need for our tasks
  require(data.table)
  require(ranger)
  require(MLmetrics)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # extract the training set subsection of case.weights
  case.weights.train = case.weights[folds$train]
  
  # build model and get prediction results
  output = rf.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                   min.node.size = doe$min.node.size[i], case.weights = case.weights.train, 
                   num.threads = num.threads)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end the cluster if it was set up
if(workers > 1)
{
  stopCluster(cl)
}

# combine the list of tables into one table
rf.cv = rbindlist(rf.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

rf.diag = rf.cv[,.(stat = factor(stat, levels = stat),
                   Multi.Log.Loss = as.vector(summary(na.omit(Multi.Log.Loss))),
                   Kappa = as.vector(summary(na.omit(Kappa))),
                   Macro.Accuracy = as.vector(summary(na.omit(Macro.Accuracy))),
                   Micro.Accuracy = as.vector(summary(na.omit(Micro.Accuracy)))),
                by = .(min.node.size)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(rf.diag)
rf.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert rf.diag into long format for plotting purposes
DT = data.table(melt(rf.diag, measure.vars = c("Multi.Log.Loss", "Kappa", "Macro.Accuracy", "Micro.Accuracy")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
rf.diag[stat == "Mean" & Kappa >= 0.35 & mod %in% rf.diag[stat == "Median" & Multi.Log.Loss <= 1.35, mod]]

# model 10 looks good
rf.diag = rf.diag[mod == 10]

# rename model to rf
rf.diag[, mod := rep("rf", nrow(rf.diag))]

# build the model
rf.mod = ranger(TripType.new ~ ., data = train.sample, min.node.size = 19,
                num.trees = 750, case.weights = case.weights,
                num.threads = 15, probability = TRUE, seed = 42)

# store model diagnostic results
rf.diag = rf.diag[,.(Multi.Log.Loss, Kappa, Macro.Accuracy, Micro.Accuracy, stat, mod)]
mods.diag = rbind(mods.diag, rf.diag)

# store the model
rf.list = list("mod" = rf.mod)
mods.list$rf = rf.list

# remove objects we no longer need
if(!(workers > 1))
{
  rm(output, Xtest, Xtrain, i, Ytest, Ytrain, case.weights.train, folds)
}

rm(rf.cv, rf.diag, rf.list, rf.mod, rf.pred, doe, DT, X, Y, diag.plot, `%fun%`,
   cl, workers, tasks, num.stats, num.rows, stat, case.weights, Y.max, num.threads)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Deep Nueral Network Model ----------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# initialize the h2o instance
h2o.init()
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# extract predictors (X) and response (Y)
X = data.table(train.sample[,!"TripType.new"])
Y = as.factor(as.numeric(train.sample$TripType.new) - 1)
Y.max = max(as.numeric(train.sample$TripType.new) - 1)

# check out the following link to understand h2o deep learning
# http://h2o-release.s3.amazonaws.com/h2o/rel-tukey/6/docs-website/h2o-docs/booklets/R_Vignette.pdf

# we have 3 hyperparameters of interest:
# hidden ~ a vector of integers indicating the number of nodes in each hidden layer
# l1 ~ L1 norm regularization to penalize large weights (may cause many weights to become 0)
# l2 ~ L2 norm regularization to penalize large weights (may cause many weights to become small)

# set up L1 & L2 penalties
l1 = 1e-5
l2 = 1e-5

# how many times the training data should be passed through the network to adjust path weights
epochs = 50

# the classes are imbalanced so lets set up the balance_classes and class_sampling_factors parameters
balance_classes = TRUE
class_sampling_factors = table(Y)
class_sampling_factors = as.vector(max(class_sampling_factors) / class_sampling_factors)

# choose the total number of hidden nodes
nodes = 150

# choose the hidden layer options to distribtuion nodes across
layers = 1:5

# choose whether to try varying structures for each layer (0 = No, 1 = Yes)
vary = 0

# initilize the size of doe
N = max(layers)
doe = matrix(ncol = N)

# build different ratios for distributing nodes across hidden layer options
for(n in layers)
{
  # single layer
  if(n == 1)
  {
    # just one layer
    op = c(1, rep(0, N - n))
    
    # store layer option
    doe = rbind(doe, op)
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op)
    
    # double layer
  } else if(n == 2)
  {
    # layers increase in size
    op1 = c(1:n, rep(0, N - n))
    # layers decrease in size
    op2 = c(n:1, rep(0, N - n))
    # layers are equal in size
    op3 = c(rep(1, length.out = n), rep(0, N - n))
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3)
    
    # largest multi-layer
  } else if(n == N)
  {
    # layers increase in size
    op1 = 1:n
    # layers decrease in size
    op2 = n:1
    # layers are equal in size
    op3 = rep(1, length.out = n)
    # layers oscilate in size, starting low
    op4 = rep(1:2, length.out = n)
    # layers oscilate in size, starting high
    op5 = rep(2:1, length.out = n)
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    op4 = op4 / sum(op4)
    op5 = op5 / sum(op5)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3, op4, op5)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3, op4, op5)
    
    # not the largest multi-layer
  } else
  {
    # op1 through op5 are the same as above
    op1 = c(1:n, rep(0, N - n))
    op2 = c(n:1, rep(0, N - n))
    op3 = c(rep(1, length.out = n), rep(0, N - n))
    op4 = c(rep(1:2, length.out = n), rep(0, N - n))
    op5 = c(rep(2:1, length.out = n), rep(0, N - n))
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    op4 = op4 / sum(op4)
    op5 = op5 / sum(op5)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3, op4, op5)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3, op4, op5)
  }
}

rm(n, N)

# remove the first row of doe becuase it was just a dummy row to append to
doe = doe[-1,]
doe = data.frame(doe)

# add cross validation ids for each scenario in doe
doe = data.frame(rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe))))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
dnn.pred = function(Xtrain, Ytrain, Xtest, Ytest, hidden, l1, l2, epochs, balance_classes, class_sampling_factors)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # make dat and Xtest into h2o objects
  dat.h2o = as.h2o(dat)
  Xtest.h2o = as.h2o(Xtest)
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         hidden = hidden,
                         l1 = l1,
                         l2 = l2,
                         epochs = epochs,
                         balance_classes = balance_classes,
                         class_sampling_factors = class_sampling_factors,
                         # activation = "Tanh",
                         # max_w2 = 10,
                         # initial_weight_distribution = "UniformAdaptive",
                         # initial_weight_scale = 0.5,
                         variable_importances = FALSE)
  
  # make predictions with the training model
  ynew = as.data.frame(predict(mod, newdata = Xtest.h2o))
  
  # ---- multi log loss metric ----
  
  # extract the probability prediction matrix
  ypred = as.matrix(ynew[,-1])
  
  # build a matrix indicating the true class values
  ytrue = model.matrix(~., data = data.frame(factor(Ytest, levels = 0:Y.max)))[,-1]
  ytrue = cbind(1 - rowSums(ytrue), ytrue)
  
  # compute the multi-class logarithmic loss
  mll = MultiLogLoss(y_pred = ypred, y_true = ytrue)
  
  # ---- Kappa ----
  
  # extract the predicted classes and actual classes
  ypred = ynew[,1]
  ytrue = factor(Ytest, levels = 0:Y.max)
  
  # build a square confusion matrix
  conf = confusion(ytrue = ytrue, ypred = ypred)
  
  # get the total number of observations
  n = sum(conf)
  
  # get the vector of correct predictions
  dia = diag(conf)
  
  # get the vector of the number of observations per class
  rsum = rowSums(conf)
  
  # get the vector of the number of predictions per class
  csum = colSums(conf)
  
  # get the proportion of observations per class
  p = rsum / n
  
  # get the proportion of predcitions per class
  q = csum / n
  
  # compute accuracy
  acc = sum(dia) / n
  
  # compute expected accuracy
  exp.acc = sum(p * q)
  
  # compute kappa
  kap = (acc - exp.acc) / (1 - exp.acc)
  
  # ---- one-vs-all metrics ----
  
  # compute a binary confusion matrix for each class
  one.v.all = lapply(1:nrow(conf), function(i)
  {
    # extract the four entries of a binary confusion matrix
    v = c(conf[i,i], 
          rsum[i] - conf[i,i], 
          csum[i] - conf[i,i], 
          n - rsum[i] - csum[i] + conf[i,i]);
    
    # build the confusion matrix
    return(matrix(v, nrow = 2, byrow = TRUE))
  })
  
  # sum up all of the matrices
  one.v.all = Reduce('+', one.v.all)
  
  # compute the micro average accuracy
  micro.acc = sum(diag(one.v.all)) / sum(one.v.all)
  
  # get the macro accuracy
  macro.acc = acc
  
  # combine all of our performance metrics
  output = data.table(Multi.Log.Loss = mll, Kappa = kap,
                    Macro.Accuracy = macro.acc, Micro.Accuracy = micro.acc)
  
  return(output)
}

# choose the number of tasks
tasks = nrow(doe)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("deep nueral network - cross validation\n")
cat(paste("task 1 started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
dnn.cv = foreach(i = 1:tasks) %do%
{
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # extract the portion of doe regarding the hidden layer structure
  doe.size = doe[,-1]
  
  # build the hidden layer structure for model i
  size = length(which(doe.size[i,] > 0))
  hidden = sapply(1:size, function(j) round(ceiling(nodes * doe.size[i,j]), 0))
  
  # build model and get prediction results
  output = dnn.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    hidden = hidden, l1 = l1, l2 = l2, epochs = epochs,
                    balance_classes = balance_classes, class_sampling_factors = class_sampling_factors)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i,])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# combine the list of tables into one table
dnn.cv = rbindlist(dnn.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

dnn.diag = dnn.cv[,.(stat = factor(stat, levels = stat),
                     Multi.Log.Loss = as.vector(summary(na.omit(Multi.Log.Loss))),
                     Kappa = as.vector(summary(na.omit(Kappa))),
                     Macro.Accuracy = as.vector(summary(na.omit(Macro.Accuracy))),
                     Micro.Accuracy = as.vector(summary(na.omit(Micro.Accuracy)))),
                  by = eval(paste0("X", seq(1:(ncol(doe) - 1))))]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(dnn.diag)
dnn.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert dnn.diag into long format for plotting purposes
DT = data.table(melt(dnn.diag, measure.vars = c("Multi.Log.Loss", "Kappa", "Macro.Accuracy", "Micro.Accuracy")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
dnn.diag[stat == "Median" & Multi.Log.Loss <= 1.5]

# model 4 looks good
dnn.diag = dnn.diag[mod == 4]

# rename model to dnn
dnn.diag[, mod := rep("dnn", nrow(dnn.diag))]

# build the hidden layer structure for model i
i = 4
doe.size = doe[,-1]
size = length(which(doe.size[i,] > 0))
hidden = sapply(1:size, function(j) round(ceiling(nodes * doe.size[i,j]), 0))
hidden

# recall the other parameters
l1
l2
epochs
class_sampling_factors

# build the model
train.sample.h2o = as.h2o(train.sample)
dnn.mod = h2o.deeplearning(y = "TripType.new",
                           x = colnames(X),
                           training_frame = train.sample.h2o,
                           hidden = c(300, 300),
                           l1 = 1e-05,
                           l2 = 1e-05,
                           epochs = 10,
                           balance_classes = TRUE,
                           class_sampling_factors = c(1, 1.817814, 1.808054, 1.613174, 1.696474, 2.453552),
                           variable_importances = FALSE)

# store model diagnostic results
dnn.diag = dnn.diag[,.(Multi.Log.Loss, Kappa, Macro.Accuracy, Micro.Accuracy, stat, mod)]
mods.diag = rbind(mods.diag, dnn.diag)

# store the model
dnn.list = list("mod" = dnn.mod)
mods.list$dnn = dnn.list

# shutdown the h2o instance
h2o.shutdown(prompt = FALSE)

# remove objects we no longer need
rm(dnn.cv, dnn.diag, dnn.list, dnn.mod, dnn.pred, i, doe, DT, output, X, Y, diag.plot, 
   layers, train.sample.h2o, epochs, hidden, l1 ,l2, nodes, size, balance_classes, class_sampling_factors,
   doe.size, Xtest, Xtrain, num.rows, num.stats, stat, tasks, Ytest, Ytrain, folds, Y.max)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Support Vector Machine Model -------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# hyperparameters of interest:
# cost ~ controls the error penalty for misclassification
  # higher values increase the error penalty and decrease the margin of seperation
  # lower values decrease the error penalty and increase the margin of seperation
# gamma ~ controls the radius of the region of influence for support vectors
  # if too large, the region of influence of any selected support vectors would only include the support vector itself and overfit the data.
  # if too small, the region of influence of any selected support vector would include the whole training set and underfit the data

# check out this link for help on tuning:
# https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur

# default value for gamma in our case:
gamma = 1 / ncol(train.sample[,!"TripType.new"])
gamma

# extract predictors (X) and response (Y)
X = data.table(train.sample[,!"TripType.new"])
Y = as.factor(as.numeric(train.sample$TripType.new) - 1)
Y.max = max(as.numeric(train.sample$TripType.new) - 1)

# create parameter combinations to test
doe = data.table(expand.grid(cost = lseq(0.001, 1000, 125),
                             gamma = gamma))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# the classes are imbalanced so lets define the class.weights parameter
class.weights = table(Y)
class.weights = setNames(as.vector(max(class.weights) / class.weights), names(class.weights))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
svm.pred = function(Xtrain, Ytrain, Xtest, Ytest, cost, gamma, class.weights)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # build the training model
  set.seed(42)
  mod = svm(y ~ .,
            data = dat,
            probability = TRUE,
            cost = cost,
            gamma = gamma,
            class.weights = class.weights)
  
  # make predictions with the training model using the test set
  ynew = predict(mod, newdata = Xtest, probability = TRUE)
  
  # ---- multi log loss metric ----
  
  # extract the probability prediction matrix
  ypred = attr(ynew, "probabilities")
  
  # build a matrix indicating the true class values
  ytrue = model.matrix(~., data = data.frame(factor(Ytest, levels = 0:Y.max)))[,-1]
  ytrue = cbind(1 - rowSums(ytrue), ytrue)
  
  # compute the multi-class logarithmic loss
  mll = MultiLogLoss(y_pred = ypred, y_true = ytrue)
  
  # ---- Kappa ----
  
  # extract the predicted classes and actual classes
  ypred = factor(as.character(ynew), levels = levels(ynew))
  ytrue = factor(Ytest, levels = 0:Y.max)
  
  # build a square confusion matrix
  conf = confusion(ytrue = ytrue, ypred = ypred)
  
  # get the total number of observations
  n = sum(conf)
  
  # get the vector of correct predictions
  dia = diag(conf)
  
  # get the vector of the number of observations per class
  rsum = rowSums(conf)
  
  # get the vector of the number of predictions per class
  csum = colSums(conf)
  
  # get the proportion of observations per class
  p = rsum / n
  
  # get the proportion of predcitions per class
  q = csum / n
  
  # compute accuracy
  acc = sum(dia) / n
  
  # compute expected accuracy
  exp.acc = sum(p * q)
  
  # compute kappa
  kap = (acc - exp.acc) / (1 - exp.acc)
  
  # ---- one-vs-all metrics ----
  
  # compute a binary confusion matrix for each class
  one.v.all = lapply(1:nrow(conf), function(i)
  {
    # extract the four entries of a binary confusion matrix
    v = c(conf[i,i], 
          rsum[i] - conf[i,i], 
          csum[i] - conf[i,i], 
          n - rsum[i] - csum[i] + conf[i,i]);
    
    # build the confusion matrix
    return(matrix(v, nrow = 2, byrow = TRUE))
  })
  
  # sum up all of the matrices
  one.v.all = Reduce('+', one.v.all)
  
  # compute the micro average accuracy
  micro.acc = sum(diag(one.v.all)) / sum(one.v.all)
  
  # get the macro accuracy
  macro.acc = acc
  
  # combine all of our performance metrics
  output = data.table(Multi.Log.Loss = mll, Kappa = kap,
                    Macro.Accuracy = macro.acc, Micro.Accuracy = micro.acc)
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 15
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("support vector machine - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
svm.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(e1071)
  require(MLmetrics)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = svm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    cost = doe$cost[i], gamma = doe$gamma[i], class.weights = class.weights)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
svm.cv = rbindlist(svm.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

svm.diag = svm.cv[,.(stat = factor(stat, levels = stat),
                     Multi.Log.Loss = as.vector(summary(na.omit(Multi.Log.Loss))),
                     Kappa = as.vector(summary(na.omit(Kappa))),
                     Macro.Accuracy = as.vector(summary(na.omit(Macro.Accuracy))),
                     Micro.Accuracy = as.vector(summary(na.omit(Micro.Accuracy)))),
                  by = .(cost, gamma)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(svm.diag)
svm.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert svm.diag into long format for plotting purposes
DT = data.table(melt(svm.diag, measure.vars = c("Multi.Log.Loss", "Kappa", "Macro.Accuracy", "Micro.Accuracy")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
svm.diag[stat == "Mean" & Multi.Log.Loss <= 1.9]

# lets go with model 22
svm.diag = svm.diag[mod == 22]

# rename model to svm as this is our chosen model
svm.diag[, mod := rep("svm", nrow(svm.diag))]

# recall class.weights
class.weights

# build our model
set.seed(42)
svm.mod = svm(TripType.new ~ ., data = train.sample, probability = TRUE, 
              cost = 0.01037837, gamma = 1/28, 
              class.weights = setNames(c(1, 1.817814, 1.808054, 1.613174, 1.696474, 2.453552), 
                                       levels(train.sample$TripType.new)))

# store model diagnostic results
svm.diag = svm.diag[,.(Multi.Log.Loss, Kappa, Macro.Accuracy, Micro.Accuracy, stat, mod)]
mods.diag = rbind(mods.diag, svm.diag)

# store the model
svm.list = list("mod" = svm.mod)
mods.list$svm = svm.list

# remove objects we no longer need
rm(gamma, svm.cv, svm.diag, svm.list, svm.mod, svm.pred, doe, DT, X, Y, diag.plot,
   workers, tasks, stat, num.stats, num.rows, class.weights, cl, Y.max)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Model Predictions ------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Models ----------------------------------------------------------------------

{

# convert mods.diag into long format for plotting purposes
DT = data.table(melt(mods.diag, id.vars = c("stat", "mod")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod, levels = unique(mod))]

# remove Inf values as these don't help
DT = data.table(DT[value < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value), fill = mod)) +
  geom_bar(stat = "identity", position = "dodge", color = "white") +
  scale_fill_manual(values = mycolors(length(levels(DT$mod)))) +
  labs(x = "Summary Statistic", y = "Value", fill = "Model") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# remove objects we no longer need
rm(diag.plot, DT)

# free memory
gc()

}

# ---- Predictions -----------------------------------------------------------------

{

# ---- gradient boosting -----------------------------------------------------------

{

# extract predictors (X), response (Y), and test set (newX)
X = as.matrix(train[, !"TripType.new"])
Y = as.numeric(train$TripType.new) - 1
newX = as.matrix(test)

# the classes are imbalanced so lets define the weight parameter where a class with more observations is weighted less
weight = table(train$TripType.new)
weight = max(weight) / weight

# make weight into a table so we can join it onto train
weight = data.table(TripType.new = names(weight), 
                    value = as.numeric(weight))

# give train an ID column so we can maintain the original order of rows
train[, ID := 1:nrow(train)]

# set TripType.new as the key column for joining purposes
setkey(train, TripType.new)
setkey(weight, TripType.new)

# join weight onto train
weight = data.table(weight[train])

# order weight by ID and extract the value column of weight
weight = weight[order(ID)]
weight = weight$value

# remove the ID column in train
train[, ID := NULL]

# build the model
set.seed(42)
mod = xgboost(label = Y, data = X,
              objective = "multi:softprob", eval_metric = "merror", nthread = 16,
              num_class = 6, eta = 0.1, max_depth = 4, nrounds = 100, min_child_weight = 7, 
              gamma = 0, subsample = 1, colsample_bytree = 1, weight = weight, verbose = 0)

# free memory
gc()

# extract the probability prediction matrix
ypred = predict(mod, newdata = newX, reshape = TRUE)

# rename the columns of the probability prediction matrix
colnames(ypred) = levels(train$TripType.new)

# extract and sort the original classes of TripType
classes = mixedsort(map.levels$TripType)

# expand the probability prediction matrix to include all of the original classes of TripType
# determine the number of tasks to do
tasks = length(classes)

# build the probability prediction matrix for all of the original classes of TripType
ynew = foreach(i = 1:tasks, .combine = "cbind") %do%
{
  # extract the new class that class i was assigned to
  new.class = map.levels[TripType == classes[i], TripType.new]
  
  # extract the number of times that class i was observed
  size.class = map.levels[TripType == classes[i], TripType.count]
  
  # extract the number of times that the new class was observed
  size.new.class = sum(map.levels[TripType.new == new.class, TripType.count])
  
  # compute the proportion of the new class that class i makes up
  prop.class = size.class / size.new.class
  
  # multiply the proportion of class i by the probability predictions for the new class to get the probability predcitions for class i
  pred.class = prop.class * ypred[,which(colnames(ypred) == new.class)]
  
  return(pred.class)
}

# rename the columns per the submission requirement
colnames(ynew) = paste0("TripType_", classes)

# the competition states that each VisitNumber may only have one TripType prediction
# so lets truncate the probability prediction matrix by averaging all of the rows for each unique VisitNumber

# extract the unique values in visit.num
unique.visit.num = unique(visit.num$value)

# choose the number of workers and tasks for parallel processing
workers = 16
tasks = length(unique.visit.num)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("truncating the probability prediction matrix\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# build the probability prediction matrix for each VisitNumber across all of the original classes of TripType
ynew = foreach(i = 1:tasks, .combine = "rbind") %dopar%
{
  # load the package we need for our tasks
  require(data.table)
  
  # extract the rows that have VisitNumber i
  row.id = visit.num[value == unique.visit.num[i], id]
  same.rows = ynew[row.id,]
  
  # if there is only one row for VisitNumber i then don't average, otherwise do so
  if(length(row.id) == 1)
  {
    # the current row is the new row
    new.row = same.rows
    
  } else
  {
    # compute the average probability prediction across all classes for VisitNumber i
    new.row = colSums(same.rows) / length(row.id)
  }
  
  # append VisitNumber i to the new probability prediction row
  new.row = c("VisitNumber" = unique.visit.num[i], new.row)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(new.row)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# export the submission
write.csv(ynew, file = "submission-nick-morris-gbm.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew, X, Y, newX, ypred, cl, classes, i, new.class, size.class,
   size.new.class, prop.class, pred.class, unique.visit.num, workers, tasks, weight)

# free memory
gc()

}

# ---- random forest ---------------------------------------------------------------

{

# the classes are imbalanced so lets define the case.weights parameter where a class with more observations is weighted less
case.weights = table(train$TripType.new)
case.weights = max(case.weights) / case.weights

# make case.weights into a table so we can join it onto train
case.weights = data.table(TripType.new = names(case.weights), 
                    value = as.numeric(case.weights))

# give train an ID column so we can maintain the original order of rows
train[, ID := 1:nrow(train)]

# set TripType.new as the key column for joining purposes
setkey(train, TripType.new)
setkey(case.weights, TripType.new)

# join case.weights onto train
case.weights = data.table(case.weights[train])

# order case.weights by ID and extract the value column of case.weights
case.weights = case.weights[order(ID)]
case.weights = case.weights$value

# remove the ID column in train
train[, ID := NULL]

# build the model
mod = ranger(TripType.new ~ ., data = train, min.node.size = 19,
             num.trees = 750, case.weights = case.weights,
             num.threads = 16, probability = TRUE, seed = 42)

# free memory
gc()

# extract the probability prediction matrix
ypred = predict(mod, data = test, num.threads = 16)$predictions

# extract and sort the original classes of TripType
classes = mixedsort(map.levels$TripType)

# expand the probability prediction matrix to include all of the original classes of TripType
# determine the number of tasks to do
tasks = length(classes)

# build the probability prediction matrix for all of the original classes of TripType
ynew = foreach(i = 1:tasks, .combine = "cbind") %do%
{
  # extract the new class that class i was assigned to
  new.class = map.levels[TripType == classes[i], TripType.new]
  
  # extract the number of times that class i was observed
  size.class = map.levels[TripType == classes[i], TripType.count]
  
  # extract the number of times that the new class was observed
  size.new.class = sum(map.levels[TripType.new == new.class, TripType.count])
  
  # compute the proportion of the new class that class i makes up
  prop.class = size.class / size.new.class
  
  # multiply the proportion of class i by the probability predictions for the new class to get the probability predcitions for class i
  pred.class = prop.class * ypred[,which(colnames(ypred) == new.class)]
  
  return(pred.class)
}

# rename the columns per the submission requirement
colnames(ynew) = paste0("TripType_", classes)

# the competition states that each VisitNumber may only have one TripType prediction
# so lets truncate the probability prediction matrix by averaging all of the rows for each unique VisitNumber

# extract the unique values in visit.num
unique.visit.num = unique(visit.num$value)

# choose the number of workers and tasks for parallel processing
workers = 16
tasks = length(unique.visit.num)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("truncating the probability prediction matrix\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# build the probability prediction matrix for each VisitNumber across all of the original classes of TripType
ynew = foreach(i = 1:tasks, .combine = "rbind") %dopar%
{
  # load the package we need for our tasks
  require(data.table)
  
  # extract the rows that have VisitNumber i
  row.id = visit.num[value == unique.visit.num[i], id]
  same.rows = ynew[row.id,]
  
  # if there is only one row for VisitNumber i then don't average, otherwise do so
  if(length(row.id) == 1)
  {
    # the current row is the new row
    new.row = same.rows
    
  } else
  {
    # compute the average probability prediction across all classes for VisitNumber i
    new.row = colSums(same.rows) / length(row.id)
  }
  
  # append VisitNumber i to the new probability prediction row
  new.row = c("VisitNumber" = unique.visit.num[i], new.row)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(new.row)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# export the submission
write.csv(ynew, file = "submission-nick-morris-rf.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew, ypred, cl, classes, i, new.class, size.class,
   size.new.class, prop.class, pred.class, unique.visit.num, workers, tasks, case.weights)

# free memory
gc()

}

# ---- deep neural network ---------------------------------------------------------

{

# initialize the h2o instance
h2o.init()
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# make train and test into h2o objects
train.h2o = as.h2o(train)
test.h2o = as.h2o(test)

# build the model
mod = h2o.deeplearning(y = "TripType.new",
                        x = colnames(test),
                        training_frame = train.h2o,
                        hidden = c(300, 300),
                        l1 = 1e-05,
                        l2 = 1e-05,
                        epochs = 10,
                        balance_classes = TRUE,
                        class_sampling_factors = c(1, 1.817814, 1.808054, 1.613174, 1.696474, 2.453552),
                        variable_importances = FALSE)

# free memory
gc()

# make predictions with the training model
ypred = as.data.frame(predict(mod, newdata = test.h2o))

# extract the probability prediction matrix
ypred = as.matrix(ypred[,-1])

# rename the columns of the probability prediction matrix
colnames(ypred) = levels(train$TripType.new)

# extract and sort the original classes of TripType
classes = mixedsort(map.levels$TripType)

# expand the probability prediction matrix to include all of the original classes of TripType
# determine the number of tasks to do
tasks = length(classes)

# build the probability prediction matrix for all of the original classes of TripType
ynew = foreach(i = 1:tasks, .combine = "cbind") %do%
{
  # extract the new class that class i was assigned to
  new.class = map.levels[TripType == classes[i], TripType.new]
  
  # extract the number of times that class i was observed
  size.class = map.levels[TripType == classes[i], TripType.count]
  
  # extract the number of times that the new class was observed
  size.new.class = sum(map.levels[TripType.new == new.class, TripType.count])
  
  # compute the proportion of the new class that class i makes up
  prop.class = size.class / size.new.class
  
  # multiply the proportion of class i by the probability predictions for the new class to get the probability predcitions for class i
  pred.class = prop.class * ypred[,which(colnames(ypred) == new.class)]
  
  return(pred.class)
}

# rename the columns per the submission requirement
colnames(ynew) = paste0("TripType_", classes)

# the competition states that each VisitNumber may only have one TripType prediction
# so lets truncate the probability prediction matrix by averaging all of the rows for each unique VisitNumber

# extract the unique values in visit.num
unique.visit.num = unique(visit.num$value)

# choose the number of workers and tasks for parallel processing
workers = 16
tasks = length(unique.visit.num)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("truncating the probability prediction matrix\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# build the probability prediction matrix for each VisitNumber across all of the original classes of TripType
ynew = foreach(i = 1:tasks, .combine = "rbind") %dopar%
{
  # load the package we need for our tasks
  require(data.table)
  
  # extract the rows that have VisitNumber i
  row.id = visit.num[value == unique.visit.num[i], id]
  same.rows = ynew[row.id,]
  
  # if there is only one row for VisitNumber i then don't average, otherwise do so
  if(length(row.id) == 1)
  {
    # the current row is the new row
    new.row = same.rows
    
  } else
  {
    # compute the average probability prediction across all classes for VisitNumber i
    new.row = colSums(same.rows) / length(row.id)
  }
  
  # append VisitNumber i to the new probability prediction row
  new.row = c("VisitNumber" = unique.visit.num[i], new.row)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(new.row)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# export the submission
write.csv(ynew, file = "submission-nick-morris-dnn.csv", row.names = FALSE)

# shutdown the h2o instance
h2o.shutdown(prompt = FALSE)

# remove objects we no longer need
rm(mod, ynew, ypred, cl, classes, i, new.class, size.class, train.h2o, test.h2o,
   size.new.class, prop.class, pred.class, unique.visit.num, workers, tasks)

# free memory
gc()

}

# ---- support vector machine ------------------------------------------------------

{

# build our model
set.seed(42)
mod = svm(TripType.new ~ ., data = train.sample, probability = TRUE, 
          cost = 0.01037837, gamma = 1/28, 
          class.weights = setNames(c(1, 1.817814, 1.808054, 1.613174, 1.696474, 2.453552), 
                                    levels(train$TripType.new)))

# free memory
gc()

# make predictions with the training model using the test set
ypred = predict(mod, newdata = test[1:1000], probability = TRUE)

# extract the probability prediction matrix
ypred = data.table(attr(ypred, "probabilities"))

# sort the columns based on the original order of levels for TripType.new
setcolorder(ypred, levels(train$TripType.new))

# make ypred into a matrix
ypred = as.matrix(ypred)

# extract and sort the original classes of TripType
classes = mixedsort(map.levels$TripType)

# expand the probability prediction matrix to include all of the original classes of TripType
# determine the number of tasks to do
tasks = length(classes)

# build the probability prediction matrix for all of the original classes of TripType
ynew = foreach(i = 1:tasks, .combine = "cbind") %do%
{
  # extract the new class that class i was assigned to
  new.class = map.levels[TripType == classes[i], TripType.new]
  
  # extract the number of times that class i was observed
  size.class = map.levels[TripType == classes[i], TripType.count]
  
  # extract the number of times that the new class was observed
  size.new.class = sum(map.levels[TripType.new == new.class, TripType.count])
  
  # compute the proportion of the new class that class i makes up
  prop.class = size.class / size.new.class
  
  # multiply the proportion of class i by the probability predictions for the new class to get the probability predcitions for class i
  pred.class = prop.class * ypred[,which(colnames(ypred) == new.class)]
  
  return(pred.class)
}

# rename the columns per the submission requirement
colnames(ynew) = paste0("TripType_", classes)

# the competition states that each VisitNumber may only have one TripType prediction
# so lets truncate the probability prediction matrix by averaging all of the rows for each unique VisitNumber

# extract the unique values in visit.num
unique.visit.num = unique(visit.num$value)

# choose the number of workers and tasks for parallel processing
workers = 16
tasks = length(unique.visit.num)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("truncating the probability prediction matrix\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# build the probability prediction matrix for each VisitNumber across all of the original classes of TripType
ynew = foreach(i = 1:tasks, .combine = "rbind") %dopar%
{
  # load the package we need for our tasks
  require(data.table)
  
  # extract the rows that have VisitNumber i
  row.id = visit.num[value == unique.visit.num[i], id]
  same.rows = ynew[row.id,]
  
  # if there is only one row for VisitNumber i then don't average, otherwise do so
  if(length(row.id) == 1)
  {
    # the current row is the new row
    new.row = same.rows
    
  } else
  {
    # compute the average probability prediction across all classes for VisitNumber i
    new.row = colSums(same.rows) / length(row.id)
  }
  
  # append VisitNumber i to the new probability prediction row
  new.row = c("VisitNumber" = unique.visit.num[i], new.row)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(new.row)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# export the submission
write.csv(ynew, file = "submission-nick-morris-rf.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew, ypred, cl, classes, i, new.class, size.class,
   size.new.class, prop.class, pred.class, unique.visit.num, workers, tasks, case.weights)

# free memory
gc()

}

}

}









