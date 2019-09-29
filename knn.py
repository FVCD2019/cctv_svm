#Import the load_iris function from datsets module
import numpy as np
import cv2

def load_DB():
    img_path = "DB/img/crop_%d.jpg"
    labels = open("DB/label.txt", "r")
    labels = labels.read().splitlines()
    labels = [int(l) for l in labels]

    imgs = []

    for i in range(150):
        img = cv2.imread(img_path % (i+1), cv2.IMREAD_GRAYSCALE)
        imgs.append(img.flatten())

    labels = np.array(labels)
    imgs = np.array(imgs)

    return imgs, labels
    

X, y = load_DB()

print(X.shape)
print(y.shape)

# splitting the data into training and test sets (80:20)
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)


#shape of train and test objects
print(X_train.shape)
print(X_test.shape)


# shape of new y objects
print(y_train.shape)
print(y_test.shape)

#import the KNeighborsClassifier class from sklearn
from sklearn.neighbors import KNeighborsClassifier

#import metrics model to check the accuracy 
from sklearn import metrics
#Try running from k=1 through 25 and record testing accuracy
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))

print(scores)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
print(knn.score(X, y))


from sklearn.externals import joblib
# Output a pickle file for the model
joblib.dump(knn, 'saved_model2.pkl') 


#0 = setosa, 1=versicolor, 2=virginica
classes = {0:'Empty',1:'Up',2:'Down'}

#Making prediction on some unseen data 
#predict for the below two random observations
x_new = X[0:5]
print(x_new.shape)

y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])
