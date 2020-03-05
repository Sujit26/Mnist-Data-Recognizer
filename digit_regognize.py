import numpy 
import nltk
import matplotlib.pyplot as pt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("digit-recognizer/train.csv").as_matrix();
print(data)
clf = DecisionTreeClassifier();
#print(clf)

xtrain = data[0:5000,1:]   #traing data
xLabel = data[0:5000,0]    #traing data label

clf.fit(xtrain,xLabel)


test_data = pd.read_csv("digit-recognizer/test.csv").as_matrix();

xtest  = test_data[1:,1:];
actualLabel = test_data[1:,0];
test_size = test_data.size();
print("test_size: ",test_size);


# p = clf.predict(xtest)
# count = 0;
# elements = 0;
# for i in range(1,20000):
#     if(p[i] == actualLabel[i]):
#         count+=1;
#         elements+=1;
#     else:
#         count = count;
#         elements+=1;
# print("accuracy: ",(count/elements)*100,"%   ,right",count,"out of ",elements)
     
#test = xtest[30]
#test.shape = (28,28)
#pt.imshow(255-test ,cmap ='gray')
#print(clf.predict( [xtest[30]] ))