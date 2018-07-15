import numpy as np
from sklearn import preprocessing, cross_validation,neighbors,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import classification_report
import scikitplot as skplt
from sklearn.neural_network import MLPClassifier
import matplotlib 
from sklearn import metrics



df=pd.read_csv('Iris.csv')
#To_load_IRIS_dataset_from_current_working_directory 

df.drop(['Id'],1,inplace=True)
#To_Drop_ID_column_from_dataframe


df.Species = pd.Categorical(df.Species)
df['label'] = df.Species.cat.codes
#TO_convert_Textual_species_name_in_"Species"_column_Into_numeric_labels_in_new_column_named_"label"



df=df.drop(['Species'],1)
#To_drop_the_Species_column_From_dataFrame

X=np.array(df.drop(['label'],1))
y=np.array(df['label'])
#To_convert_and_store_data_into_numpy_array_object_X





X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.3)
#To_split_data_and_labels_into_test_and_train

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100), random_state=10,learning_rate='adaptive')
#To_define_classifier_and_set_its_parameters


clf.fit(X_train,y_train)
#To_train_the_models


accuracy=clf.score(X_test,y_test)
print("accuracy of the model is :",accuracy)
#to calculate and show the accuracy of the model

prediction=clf.predict(X_test)


print(y_test)
print(prediction)
#to get the predicted value for X_test samples



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#method to print confusion metrix



cm = confusion_matrix(y_test,prediction)
np.set_printoptions(precision=3)

cm_plot_labels = ['Iris-setosa','Iris-versicolor','Iris-virginica']


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=cm_plot_labels, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=cm_plot_labels , normalize=True, title='Normalized confusion matrix')


#plt.show()

'''
y_prob = clf.predict_proba(X_test)
plt.figure()
#to calculate the prediction probability of X_test samples
skplt.metrics.plot_roc_curve(y_test, y_prob)
'''
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(y_test,prediction, target_names=target_names))

plt.show()



