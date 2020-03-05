from PIL import Image
import numpy 
import nltk
import matplotlib.pyplot as pt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import time

data = pd.read_csv("digit-recognizer/train.csv").as_matrix();
clf = DecisionTreeClassifier();

print(data.size/785);

xtrain = data[0:2000,1:]   #traing data
xLabel = data[0:2000,0]    #traing data label
clf.fit(xtrain,xLabel)


xtest  = data[1:,1:]
actualLabel = data[1:,0]

p = clf.predict(xtest)
count = 0;
elements = 0;

for i in range(1,p.size):
    if(p[i] == actualLabel[i]):
        count+=1;
    else:
        count = count;
    elements+=1;
    #print(clf.score(xtest[i],actualLabel[i]));
##print("accuracy: ",(count/elements)*100,"% ,right",count,"out of ",elements)
test = xtest[332]
test.shape = (28,28)
pt.imshow(255-test ,cmap ='gray')
###################################################

columnNames = list()
for i in range(784):
    pixel = 'pixel'
    pixel += str(i)
    columnNames.append(pixel)
train_data = pd.DataFrame(columns = columnNames)
start_time = time.time()


img_name = '1.png'
img = Image.open(img_name)
print(img)
rawData = img.load()
        #print rawData
data = []
for y in range(28):
    for x in range(28):
        data.append(rawData[x,y][0])
    k = 0
    #print data
    train_data.loc[i] = [data[k] for k in range(784)]
    #print train_data.loc[0]
####################################################
print(clf.predict( [xtest[332]] ))