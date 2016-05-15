import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
def loadDataset(filename,split,data=[],target=[]):
    with open(filename,'rb') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)):
            for y in range(58):
                dataset[x][y]=float(dataset[x][y])
        for x in range(len(dataset)):
                data.append(dataset[x][:2])           
                target.append(dataset[x][-1]) 
# import some data to play with
data=[]
target=[]
split=0.67
loadDataset('spambase.data',split,data,target)
X = data  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = target

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC().fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
Nu_svc=svm.NuSVC().fit(X,y)
zero=[]
one=[]

for i in data:
    zero.append(i[0])
    one.append(i[1])
#print zero
#print one   
   

# create a mesh to plot in
x_min, x_max = min(zero), max(zero) 
y_min, y_max = min(one) , max(one)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC ',
          'SVC with RBF kernel','LinearSVC (linear kernel)',
          'NuSVC']


for i, clf in enumerate((svc,rbf_svc, lin_svc,Nu_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(zero, one, c=y, cmap=plt.cm.Paired)
    plt.xlabel('make')
    plt.ylabel('address')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
