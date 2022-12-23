from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def PCA_calculation(data, features):


    scaler = StandardScaler()
    scaler.fit(data)
    scaler_PCA = scaler.transform(data)
    pca = PCA(n_components = data.shape[1])
    pca.fit_transform(scaler_PCA)
    PCs =  ['PC' + str(i) for i in range(1,len(features) + 1)]
    values = pca.explained_variance_ratio_.tolist()
    PCs_values = dict(zip(PCs, values))

    return pca, PCs_values

def numberlist(nums,limit):   
    sum=0  
    for index,i in enumerate(nums):  
        sum += i
        if sum>=limit:  
            return nums[:index+1]
        
def PCA_feature_importance_selection(pca, features, variance_explained):

    importance = abs(pca.components_)
    values = pca.explained_variance_ratio_.tolist()
    features_array = np.zeros(len(features), dtype =np.float64)
    for i in range(len(features)):
        temp = np.zeros(len(features), dtype =np.float64)
        for j in range(len(features)):
            single_PC = importance[0]*values[j]
            temp = temp + single_PC
        features_array = features_array + temp
    idx = (-features_array).argsort()[:len(features)]

    feature_importance = []
    for i in range(len(idx)):
        feature = features[idx[i]]
        feature_importance.append(feature)
    sorted_array = np.sort(features_array)
    reverse_array = sorted_array[::-1]
    reverse_list = reverse_array.tolist()
    normalized_scores = []
    for r in range(len(reverse_list)):
        elem = (reverse_list[r]/sum(reverse_list))*100
        normalized_scores.append(elem)
    limit = sum(reverse_list)*variance_explained 
    threshold = normalized_scores[len(numberlist(reverse_list,limit))-1]

    feature_with_score_list = dict(zip(feature_importance, normalized_scores))
    feature_with_score_list = dict(sorted(feature_with_score_list.items()))

    d = dict((k, v) for k, v in feature_with_score_list.items() if v > threshold)
    selected_features = list(d.keys())

    return normalized_scores, feature_importance, selected_features,threshold



def PCA_explained_variance(pca, features):

    explained_variance_ratio = pca.explained_variance_ratio_*100
    explained_variance_ratio = explained_variance_ratio.tolist()
    variance_exp_cumsum = pca.explained_variance_ratio_.cumsum().round(2)*100
    PCs = [ 'PC'+str(i) for i in range(1,len(features)+1)]

    return PCs, explained_variance_ratio, variance_exp_cumsum