library(dplyr)
library(mlr)
library(xtable)
library(caret)
library(knitr)
library(FSelector)
library(kernlab)
library(gbm)
trainCsv<-
  "http://madeyski.e-informatyka.pl/download/stud/CaseStudies/Lucene/lucene2.2train.csv"
testCsv<-
  "http://madeyski.e-informatyka.pl/download/stud/CaseStudies/Lucene/lucene2.4test.csv"
trainOrig<-read.csv(trainCsv, header=TRUE, sep=",")
testOrig<-read.csv(testCsv, header=TRUE, sep=",")
head(trainOrig, n=2)


mlr::summarizeColumns(trainOrig) #gives a comprehensive view of the data se
summary(trainOrig)

mlr::summarizeColumns(testOrig)
summary(testOrig)

par(mfrow=c(1,4))
for(i in 5:8) {
  boxplot(trainOrig[,i], main=names(trainOrig)[i])
}

# split input (IVs) and output (DV)
iv <- trainOrig[,5:8] # noc, cbo, wmc, dit...
dv <- trainOrig[,29] # isBuggy
caret::featurePlot(x=iv,y=as.factor(dv),plot="box",
                   scales=list(x=list(relation="free"),y=list(relation="free")),
                   auto.key=list(columns=2))


iv <- trainOrig[,23:26] # #nr, ndc, nml, ndpv...
dv <- trainOrig[,29] # isBuggy
caret::featurePlot(x=iv, y=as.factor(dv), plot="box", auto.key=list(columns=2),
                   scales=list(x=list(relation="free"), y=list(relation="free")))

percentage <- prop.table(table(trainOrig$isBuggy)) * 100
cbind(freq=table(trainOrig$isBuggy), percentage=percentage)





train <- trainOrig %>% dplyr::select(-c(X, Project, Version, Class))
mlr::summarizeColumns(train) %>% xtable::xtable() %>%
  print(booktabs = TRUE, include.rownames=FALSE, scalebox = 0.74, NA.string = "NA")

test <- testOrig %>% dplyr::select(-c(X, Project, Version, Class))
mlr::summarizeColumns(test) %>% xtable::xtable() %>%
  print(booktabs = TRUE, include.rownames=FALSE, scalebox = 0.74, NA.string = "NA")

imp <- mlr::impute(train,
                   classes = list( #Named list containing imputation techniques for classes of data
                     factor = mlr::imputeMode(), integer = mlr::imputeMean(), numeric = mlr::imputeMean() ) )
train <- imp$data
mlr::summarizeColumns(train) %>% xtable::xtable() %>%
  print(booktabs = TRUE, include.rownames=FALSE, scalebox = 0.7, NA.string = "NA") # Show column summary

imp <- mlr::impute(test,
                   classes = list( #Named list containing imputation techniques for classes of data
                     factor = mlr::imputeMode(), integer = mlr::imputeMean(), numeric = mlr::imputeMean() ) )
test <- imp$data
mlr::summarizeColumns(test) %>% xtable::xtable() %>%
 print(booktabs = TRUE, include.rownames=FALSE, scalebox = 0.7, NA.string = "NA") # Show column summary

print(mlr::listLearners("regr", check.packages = TRUE,
                        properties = c("factors", "missings")
)[c("class", "package")], max.print=12)

trainTask<-mlr::makeClassifTask(data=train,target="isBuggy",positive="TRUE")
testTask<-mlr::makeClassifTask(data=test, target="isBuggy",positive="TRUE")

print(trainTask)
print(testTask)

featureImportance<-mlr::generateFilterValuesData(testTask,
                                                 method = "FSelector_information.gain")
mlr::plotFilterValues(featureImportance)





# create a Logistic Regression learner (i.e., specify a ML algorithm)
logisticRegression.learner <- mlr::makeLearner("classif.logreg", predict.type = "response")
# train the learner on the training task (set)
logisticRegression.model <- mlr::train(logisticRegression.learner, task = trainTask)
# predict values of the DV on a test task (set)
pred = predict(logisticRegression.model, task = testTask)
# Evaluate the learner via the mean misclassification error (mmcse), mcc, acc, f1, kappa
perfMeasures <- mlr::performance(pred, measures = list(mlr::mcc,mlr::mmce,mlr::acc,mlr::f1,mlr::kappa))
# perf. measures: mcc=(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
# mean misclassification error mmce=mean(response != truth)
perfMeasures


# create a Quadratic Discriminant Analysis model
quadraticDiscriminantAnalysis.learner <- mlr::makeLearner("classif.qda", predict.type = "response")
# train the learner on the training task (set)
quadraticDiscriminantAnalysis.model <- mlr::train(quadraticDiscriminantAnalysis.learner,
                                                  task = trainTask)
# predict values of the DV on a test task (set)
pred = predict(quadraticDiscriminantAnalysis.model, task = testTask)
# Evaluate the learner via the mean misclassification error (mmcse), mcc, acc, f1, kappa
perfMeasures<-mlr::performance(pred, measures = list(mlr::mcc, mlr::mmce, mlr::acc, mlr::f1, mlr::kappa))
perfMeasures

confusionMatrixQDA<-mlr::calculateConfusionMatrix(pred)
confusionMatrixQDA

# create a decision tree that should capture non-linear relations better than a LR model
decisionTree.learner <- makeLearner("classif.rpart", predict.type = "response")
# train the learner on the training task (set)
decisionTree.model <- mlr::train(decisionTree.learner, task = trainTask)
# predict values of the DV on a test task (set)
pred = predict(decisionTree.model, task = testTask)
# Evaluate the learner via the mean misclassification error (mmcse), mcc, acc, f1, kappa
perfMeasures<-mlr::performance(pred, measures = list(mlr::mcc, mlr::mmce, mlr::acc, mlr::f1, mlr::kappa))
perfMeasures

confusionMatrixRpart<-mlr::calculateConfusionMatrix(pred)
confusionMatrixRpart

confusionMatrixLR
confusionMatrixQDA
confusionMatrixRpart

getParamSet("classif.rpart")

set.seed(1) # set seed to enable reproducibility
# create a decision tree that should capture non-linear relations better than a LR model
decisionTree.learner <- mlr::makeLearner("classif.rpart", predict.type = "response")
# set 3 fold cross validation
setCV <- mlr::makeResampleDesc("CV",iters = 3L)
# search for hyperparameters
paramSet <- ParamHelpers::makeParamSet(
  #min.num. of obs. in a node for a split to take place
  ParamHelpers::makeIntegerParam("minsplit",lower = 10, upper = 50),
  #min.num, of obs. in terminal nodes 5..50
  ParamHelpers::makeIntegerParam("minbucket", lower = 5, upper = 50),
  #the complexity parameter (cp)
  ParamHelpers::makeNumericParam("cp", lower = 0.001, upper = 0.3)
)
# hypertune the parameters
tuneParams <- mlr::tuneParams(learner = decisionTree.learner, resampling = setCV,
                              task = trainTask, par.set = paramSet, control = mlr::makeTuneControlGrid(),
                              measures = mlr::mcc) ##do a grid search
# check best parameter
tuneParams$x

# using hyperparameters for modeling
decisionTreeTuned.learner <- setHyperPars(decisionTree.learner, par.vals = tuneParams$x)
# train the learner on the training task (set)
decisionTreeTuned.model <- mlr::train(decisionTreeTuned.learner, task = trainTask)
# predict values of the DV on a test task (set)
pred = predict(decisionTreeTuned.model, task = testTask)
# Evaluate the learner via the mean misclassification error (mmcse), mcc, acc, f1, kappa
perfMeasures<-mlr::performance(pred, measures = list(mlr::mcc, mlr::mmce, mlr::acc, mlr::f1, mlr::kappa))
perfMeasures

confusionMatrixTunedRpart<-mlr::calculateConfusionMatrix(pred)
confusionMatrixTunedRpart

confusionMatrixRpart
confusionMatrixTunedRpart

# create a decision tree that should capture non-linear relations better than a LR model
decisionTree.learner <- makeLearner("classif.rpart", predict.type = "response")
# set 3 fold cross validation
setCV <- makeResampleDesc("CV",iters = 3L)
# search for hyperparameters
paramSet <- makeParamSet(
  makeIntegerParam("minsplit",lower = 10, upper = 50), #min.num. of obs. in a node for a split to take place
  makeIntegerParam("minbucket", lower = 5, upper = 50), #min.num, of obs. in terminal nodes 5..50
  makeNumericParam("cp", lower = -10, upper = 0, trafo = function(x) 2^x) #the complexity parameter (cp).
)
# hypertune the parameters
tuneParams <- tuneParams(learner = decisionTree.learner, resampling = setCV, task = trainTask,
                         par.set = paramSet, control = makeTuneControlGrid(), measures = mlr::mcc) ##do a grid search
# check best parameter
tuneParams$x

# using hyperparameters for modeling
decisionTreeTuned.learner <- setHyperPars(decisionTree.learner, par.vals = tuneParams$x)
# train the learner on the training task (set)
decisionTreeTuned.model <- mlr::train(decisionTreeTuned.learner, task = trainTask)
# predict values of the DV on a test task (set)
pred = predict(decisionTreeTuned.model, task = testTask)
# Evaluate the learner via the mean misclassification error (mmcse), mcc, acc, f1, kappa
perfMeasures<-mlr::performance(pred, measures = list(mlr::mcc, mlr::mmce, mlr::acc, mlr::f1, mlr::kappa))
perfMeasures

confusionMatrix<-mlr::calculateConfusionMatrix(pred)
confusionMatrix

getParamSet("classif.randomForest")

set.seed(1) # set seed to enable reproducibility
# create a Random Forest that often produce top results as it is an ensemble of trees
# it averages the prediction given by each tree and produces a generalized result
randomForest.learner <- makeLearner("classif.randomForest", predict.type="response") #, par.vals = list(ntree = 200, mtry = 3))
setCV <- makeResampleDesc("CV",iters = 3L)# set 3 fold cross validation
# search for hyperparameters
paramSet <- makeParamSet(
  #Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times./lower=50
  makeIntegerParam("ntree",lower=40, upper=500),
  #Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3) /upper=10/12
  makeIntegerParam("mtry", lower=1, upper=20),
  #Minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time). Note that the default values are different for classification (1) and regression (5). /upper=50/60
  makeIntegerParam("nodesize", lower=1, upper=60)
)
# hypertune the parameters
tuneParams <- tuneParams(learner = randomForest.learner, resampling = setCV,
                         task = trainTask, par.set = paramSet, control = makeTuneControlRandom(maxit = 70L),
                         measures = mlr::mcc) # Do a random search. It is faster than grid search, but sometimes it turns out to be less efficient. In grid search, the algorithm tunes over every possible combination of parameters provided. In a random search, we specify the number of iterations and it randomly passes over the parameter combinations. In this process, it might miss out some important combination of parameters which could have returned maximum accuracy
# check best parameters
tuneParams$x

# using hyperparameters for modeling
randomForestTuned.learner <- setHyperPars(randomForest.learner, par.vals = tuneParams$x)
# train the learner on the training task (set)
randomForestTuned.model <- mlr::train(randomForestTuned.learner, task = trainTask)
# predict values of the DV on a test task (set)
pred = predict(randomForestTuned.model, task = testTask)
# Evaluate the learner via the mean misclassification error (mmcse), mcc, acc, f1, kappa
perfMeasures<-mlr::performance(pred, measures = list(mlr::mcc, mlr::mmce, mlr::acc, mlr::f1, mlr::kappa))
perfMeasures


confusionMatrixTunedRF<-mlr::calculateConfusionMatrix(pred)
confusionMatrixTunedRF

getParamSet("classif.ksvm")

set.seed(1) # set seed to enable reproducibility
# create a Random Forest that often produce top results as it is an ensemble of trees
# it averages the prediction given by each tree and produces a generalized result
ksvm.learner <- makeLearner("classif.ksvm", predict.type = "response")
# set 3 fold cross validation
setCV <- makeResampleDesc("CV",iters = 3L)
# search for hyperparameters
paramSet <- makeParamSet(
  makeDiscreteParam("C", values=2^c(-8,-4,-2,0,1,2)), #cost parameter
  makeDiscreteParam("sigma", values=2^c(-8,-6,-4,-2,0,2,4))#impacts the smoothness of the decision boundary (RBF Kernel Parameter) c(-8,-4,0,4)
)
# hypertune the parameters
tuneParams <- tuneParams(learner = ksvm.learner, resampling = setCV, task = trainTask,
                         par.set = paramSet, control = makeTuneControlGrid(), measures = mlr::mcc) # Do a grid search over every
#possible combination of parameters provided.
# check best parameters
tuneParams$x

# using hyperparameters for modeling
ksvmTuned.learner <- setHyperPars(ksvm.learner, par.vals = tuneParams$x)
# train the learner on the training task (set)
ksvmTuned.model <- mlr::train(ksvmTuned.learner, task = trainTask)
# predict values of the DV on a test task (set)
pred = predict(ksvmTuned.model, task = testTask)
# Evaluate the learner via the mean misclassification error (mmcse), mcc, acc, f1, kappa
perfMeasures<-mlr::performance(pred, measures = list(mlr::mcc, mlr::mmce, mlr::acc, mlr::f1, mlr::kappa))
perfMeasures


confusionMatrixTunedSVM<-mlr::calculateConfusionMatrix(pred)
confusionMatrixTunedSVM

confusionMatrixTunedRpart

confusionMatrixTunedRF

confusionMatrixTunedSVM

getParamSet("classif.gbm")

set.seed(1) # set seed to enable reproducibility
gbm.learner <- makeLearner("classif.gbm", predict.type = "response") # create GBM
setCV <- makeResampleDesc("CV", iters=3L) # set 3 fold cross validation
# search for hyperparameters
paramSet <- makeParamSet(
  #if not specified, gbm will guess: if DV has only 2 unique values, bernoulli is assumed; otherwise, if the response is a factor, multinomial is assumed; otherwise, if the response has class "Surv", coxph is assumed; otherwise, gaussian is assumed.
  makeDiscreteParam("distribution", values = "bernoulli"),
  makeIntegerParam("n.trees", lower = 100, upper = 1000), #number of trees
  makeIntegerParam("interaction.depth", lower = 1, upper = 10),#1:additive model, 2:two-way interactions, etc (depth of tree)
  makeIntegerParam("n.minobsinnode", lower = 5, upper = 20), #min. total weight needed in each node
  makeNumericParam("shrinkage",lower = 0.001, upper = 0.5)#shrinkage or learning rate, 0.001-0.1 usually works
)
# hypertune the parameters
tuneParams <- tuneParams(learner = gbm.learner, resampling = setCV, task = trainTask,
                         par.set = paramSet, control = makeTuneControlRandom(maxit = 150L), measures = mlr::mcc) # Do a random search.
tuneParams$x # check best parameters

# using hyperparameters for modeling
gbmTuned.learner <- setHyperPars(gbm.learner, par.vals = tuneParams$x)
# train the learner on the training task (set)
gbmTuned.model <- mlr::train(gbmTuned.learner, task = trainTask)
# predict values of the DV on a test task (set)
pred = predict(gbmTuned.model, task = testTask)
# Evaluate the learner via the mean misclassification error (mmcse), mcc, acc, f1, kappa
perfMeasures<-mlr::performance(pred, measures = list(mlr::mcc, mlr::mmce, mlr::acc, mlr::f1, mlr::kappa))
perfMeasures

confusionMatrixTunedGBM<-mlr::calculateConfusionMatrix(pred)
confusionMatrixTunedGBM

confusionMatrixTunedRpart
confusionMatrixTunedRF
confusionMatrixTunedGBM

