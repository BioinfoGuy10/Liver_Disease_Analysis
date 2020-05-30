import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from numpy import pi, sin, cos
import plotly.graph_objects as go

###################Read the files##############################
liver_disease = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv", delimiter=",")
pd.DataFrame(liver_disease)
#########################Let us check for any missing value and fill them accordingly###########
print(liver_disease.isnull().sum())

##########We can see that there are 4 missing values in Albumin-Globulin Ratio#########
liver_disease["Globulin"]=liver_disease["Total_Proteins"]-liver_disease["Albumin"]#Find the globulin levels
liver_disease["A_G_Ratio"]=round(liver_disease["Albumin"]/liver_disease["Globulin"],2)#Create a new Albumin-globulin ratio column
liver_disease=liver_disease.drop(['Albumin_Globulin_Ratio'], axis=1)#Drop the columns with the missing values

Age_Group=[]
##################Let us get started with EDA#################################
############First, I want to see what age group are most affected by this disease##############
for i in range(0, liver_disease.shape[0]):#Lets create an age group
    if(liver_disease["Age"][i]<=10):
        Age_Group.append("1-10")
    elif(liver_disease["Age"][i]>10 and liver_disease["Age"][i]<=20):
        Age_Group.append("10-20")
    elif(liver_disease["Age"][i]>20 and liver_disease["Age"][i]<=30):
        Age_Group.append("20-30")
    elif(liver_disease["Age"][i]>30 and liver_disease["Age"][i]<=40):
        Age_Group.append("30-40")
    elif(liver_disease["Age"][i]>40 and liver_disease["Age"][i]<=50):
        Age_Group.append("40-50")
    elif(liver_disease["Age"][i]>50 and liver_disease["Age"][i]<=60):
        Age_Group.append("50-60")
    else:
        Age_Group.append("60+")

liver_disease["Age-Group"]=Age_Group
liver_disease=liver_disease.sort_values(by=["Age-Group"])
plt.hist(liver_disease["Age-Group"].loc[(liver_disease["Diagnosis"]==1)], bins=7, histtype="stepfilled",edgecolor="black")
plt.title("Disease by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.savefig("Age-Group-diagnosis.png") #We can thus conclude that Liver disease predominantly affects older people

########################Let us begin with model fitting and feature selection###############
########################Logistic Regression###########################

dependent = liver_disease["Diagnosis"]
######################Remove features that are collinear or redundant################     
attributes = liver_disease.drop(['Diagnosis', 'Age-Group'], axis=1)

############################Perform feature selection###################

#######################Recode gender values to use Logistic Regression########
attributes["Gender"]=attributes["Gender"].replace("Male",1)
attributes["Gender"]=attributes["Gender"].replace("Female",2)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(attributes, dependent)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(attributes.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

##############Take the top five features#########################
best_cols = list(featureScores.sort_values(by=['Score'],ascending=False)["Specs"])
main_data = attributes.loc[:,best_cols[0:4]]
######################Split the dataset into train and test######################
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split( 
        main_data, dependent, test_size = 0.25, random_state = 1)

######################Train the data on the train set##################
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(penalty="l2",solver="liblinear",random_state = 0, max_iter=1000) 
classifier.fit(xtrain, ytrain) 

#######################Let us perform prediction using the test data##########
y_pred = classifier.predict(xtest)

##################Test the performance of our model-Confusion Matrix##############
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 
  
print ("Confusion Matrix : \n", cm) 

################Performance Measure-Accuracy#################
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred)) 
from sklearn.model_selection import cross_val_score

# Recall
from sklearn.metrics import recall_score
print("Recall :",recall_score(ytest, y_pred))

# Precision
from sklearn.metrics import precision_score
print("Precision: ",precision_score(ytest, y_pred))
scores = cross_val_score(classifier, xtest, ytest, cv=30)
print('Cross-Validation Accuracy Scores', scores)


# filter out the patients
patients = liver_disease.loc[(liver_disease["Diagnosis"]==1)]
patients_feature_1_Male=patients.loc[:,best_cols[0]].loc[(patients["Gender"]=="Male")]
patients_feature_2_Male=patients.loc[:,best_cols[1]].loc[(patients["Gender"]=="Male")]
patients_feature_1_Female=patients.loc[:,best_cols[0]].loc[(patients["Gender"]=="Female")]
patients_feature_2_Female=patients.loc[:,best_cols[1]].loc[(patients["Gender"]=="Female")]
                            
# filter out the healthy
healthy = liver_disease.loc[(liver_disease["Diagnosis"]==2)]
healthy_feature_1_Male=healthy.loc[:,best_cols[0]].loc[(healthy["Gender"]=="Male")]
healthy_feature_2_Male=healthy.loc[:,best_cols[1]].loc[(healthy["Gender"]=="Male")]
healthy_feature_1_Female=healthy.loc[:,best_cols[0]].loc[(healthy["Gender"]=="Female")]
healthy_feature_2_Female=healthy.loc[:,best_cols[1]].loc[(healthy["Gender"]=="Female")]

# plots for Male
fig = plt.figure()
ax1 = fig.add_subplot()
#ax1.xlim(1,2900)
ax1.scatter(patients_feature_1_Male, patients_feature_2_Male,s=4,linewidths=0, label='Diseased')
ax1.scatter(healthy_feature_1_Male, healthy_feature_2_Male,s=4, linewidths=0,label='Healthy')
ax1.legend()
ax1.set_title("Diseased Vs Healthy Male")
ax1.set_xlabel(best_cols[0])
ax1.set_ylabel(best_cols[1])

plt.savefig(best_cols[0]+" and "+best_cols[1]+" Male Separation.png")

# plots for Female
fig = plt.figure()
ax1 = fig.add_subplot()
#ax1.xlim(1,2900)
ax1.scatter(patients_feature_1_Female, patients_feature_2_Female,s=4,linewidths=0, label='Diseased')
ax1.scatter(healthy_feature_1_Female, healthy_feature_2_Female,s=4, linewidths=0,label='Healthy')
ax1.legend()
ax1.set_title("Diseased Vs Healthy Female")
ax1.set_xlabel(best_cols[0])
ax1.set_ylabel(best_cols[1])

plt.savefig(best_cols[0]+" and "+best_cols[1]+" Female Separation.png")

#####################Finally, lets try PCA with the top 5 features##################
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
collinear_cols= ["Total_Bilirubin", "Direct_Bilirubin", "Total_Proteins", "Albumin", "Globulin", "A_G_Ratio"]
principalComponents = pca.fit_transform(main_data)
principalDF = pd.DataFrame(data= principalComponents, columns= ['Principal Component 1', "Principal Component 2"])
print("The number of rows in PrincipalDF is {}". format(principalDF.shape))
frames = [principalDF, dependent]
finalDf = pd.concat(frames, axis=1)
fig= plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)
 
targets = [1, 2]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Diagnosis'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1'],
               finalDf.loc[indicesToKeep, 'Principal Component 2'],
               c = color,
               s = 50
               )
ax.legend(targets)
ax.grid()

plt.savefig('PCA_plot.png')
