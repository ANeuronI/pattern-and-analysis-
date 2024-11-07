from sklearn.datasets import load_breast_cancer 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

breast = load_breast_cancer() 
breast_data = breast.data
print("Data Shape:",breast_data.shape) 
breast_labels = breast.target

labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1) 
print("Final Data Shape:",final_breast_data.shape)

breast_dataset = pd.DataFrame(final_breast_data) 
features = breast.feature_names
print("Features: \n",features)
features_labels = np.append(features,'label')
breast_dataset.columns = features_labels 
print("First few Rows \n: ",breast_dataset.head())
breast_dataset['label'] = breast_dataset['label'].replace(0, 'Benign')
breast_dataset['label'] = breast_dataset['label'].replace(1, 'Malignant')
print("Last few Rows:", breast_dataset.tail())
#Data Visualizing using PCA

x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features print("Mean of normalised data:",np.mean(x))
print("Standard Devaition of normalised data:",np.std(x)) 
feat_cols = ['feature'+str(i) for i in range(x.shape[1])] 
normalised_breast = pd.DataFrame(x,columns=feat_cols)
print("Data after normalised: \n",normalised_breast.tail())

pca_breast = PCA(n_components=2) 
principalComponents_breast = pca_breast.fit_transform(x) 
principal_breast_Df = pd.DataFrame(data =

principalComponents_breast,columns=['principal component 1', 'principal component 2'])
print("Data after PCA applied: \n",principal_breast_Df.tail()) 
print('Explained variation per principal component:{}'.format(pca_breast.explained_variance_ratio_))

plt.figure(figsize=(10,10)) 
plt.xticks(fontsize=12) 
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20) 
targets = ['Benign', 'Malignant']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target 
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1'],principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
    plt.legend(targets,prop={'size': 15}) 
plt.show()