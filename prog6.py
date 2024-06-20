#WITH SKLEARN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("./glass.csv")

# Prepare data
X = df.drop("Type", axis=1).values
y = df['Type'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Train and predict using scikit-learn's KNN
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, Y_train)
predictions = knn_clf.predict(X_test)

# Accuracy of scikit-learn's KNN
print("scikit-learn KNN Predictions:", predictions)
print("scikit-learn KNN Accuracy:", accuracy_score(y_pred=predictions, y_true=Y_test))

# Plotting
plt.scatter(X[:, 2], X[:, 3], c=y, cmap='viridis')
plt.title('Scatter plot of the glass dataset')
plt.xlabel('Feature 2')
plt.ylabel('Feature 3')
plt.colorbar(label='Type')
plt.show()
'''WITHOUT SKLEARN
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def ec(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

df=pd.read_csv("./glass.csv")
df.head()
from collections import Counter
class KNN:
    def __init__(self,k=3):
        self.k=k

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y 

    def predict(self,X):
        predictions=[self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        #Compute distance from one given point to all the points in X_train
        distances=[ec(x1=x,x2=x_train) for x_train in self.X_train]

        #Get k closest indices and labels
        k_indices=np.argsort(distances)[:self.k]
        k_labels=[self.y_train[i] for i in k_indices]

        #Get most common class label
        co=Counter(k_labels).most_common()
        return co[0][0]

    
X=df.drop("Type",axis=1).values
y=df['Type'].values
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=40)
clf=KNN(k=3)
clf.fit(X_train,Y_train)
predictions=clf.predict(X_test)
print(predictions)
plt.scatter(X[:,2],X[:,3],c=y)
plt.show()

print(accuracy_score(y_pred=predictions,y_true=Y_test))

'''