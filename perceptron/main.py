from adlineGB import AdlineGD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v') # square, x, o, triangle_down, triangle_up
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') # cyan 青色
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max =X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))# arange(return ndarray) ,range (return object)
    print(xx1)
    print(xx1.ravel())
    print(np.array([xx1.ravel(), xx2.ravel()]).T) # np,array : [[],[]]
    Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # ravel表示将矩阵展开成一维，flatten功能相同，但是复制一份不影响原矩阵， T表示转置
    Z=Z.reshape(xx1.shape)  #reshape([])将一维数组重组成矩阵
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha= 0.8, c=cmap(idx), marker=markers[idx], label=cl)
def main () :
    df = pd.read_csv('/Users/yangchong/PycharmProjects/iris.data', header= None)
    print(df.tail())
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    print(y)
    X = df.iloc[0:100, [0, 2]].values
    print(X)
    '''
    plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='o', label='setosa')
    plt.scatter(X[50:, 0], X[50:100, 1], color='red', label='versicolor', marker='x')
    plt.xlabel('petal length')
    plt.xlabel('sepal length')
    plt.legend(loc ='upper left')
    plt.show()
    '''
    ppn = Perceptron(eta=0.01, n_iter=10)
    ppn.fit(X, y)
    '''
    plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()
    
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))# 绘制1*2个子图，
    ada1 = AdlineGD(eta=0.0001, n_iter=10).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error')
    ax[0].set_title('Adaline-learning rate=0.01')
    plt.show()
    '''
    X_std = np.copy(X) # 标准化：x-mean/std，numpy的方法，标准差: 标准化后均值为0，方差为1 (正太分布)
    X_std[:, 0] = (X[:, 0]-X[:, 0].mean())/X[:, 0].std()
    X_std[:, 1] = (X[:, 1]-X[:, 1].mean())/X[:, 1].std()

    print('xx', X_std)

    arr = X_std.ravel()
    print(arr)
    print(arr[10])

if __name__ == '__main__':
    main()
