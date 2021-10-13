import array as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#loading the dataset
url= "https://bit.ly/w-data"
data=pd.read_csv(url)
print("Data imported succesfully ")

data.head(12)
data.describe()
 data.isnull().sum()
#plotting the distribution of scores 
data.plot(x="Hours", y="Scores", style="o")
plt.title("No.of Hours studied vs scores")
plt.xlabel("No. of hours studied ")
plt.ylabel("Scores")
plt.show()
#dividing data into "attributes"(inputs) & "labels"(outputs)
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
#splitting the data into training and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train,y_train)

print("Training completed successfully")
#Plotting the regression line
line=regr.coef_*x+regr.intercept_
plt.scatter(x,y)
plt.show()
print(x_test)
#Predicting the scores
y_predicted=regr.predict(x_test)
print(regr.predict(x_test))
d=[x_test,y_test,y_predicted]
d
df=pd.DataFrame({"Actual Value":y_test,"Predicted value":y_predicted})
df
#predicting score if a student studies for 9.25 hours/day
Study_hours= 9.25
predicted_score= regr.predict([[Study_hours]])
print("Number of Hours= {}".format(Study_hours))
print("Predicted Score: {}".format(predicted_score[0]))
print( "If a student studies for 9.25 hours/day, his score would be:",predicted_score)
from sklearn import metrics 
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_predicted)) 
