from dash import  Dash, dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import os, base64, pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
from sklearn.decomposition import PCA

import xgboost as xgb
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

import xgboost as xgb
import pca_module.functions as pca_func

pwd = os.getcwd()

### Load images
scheme_filename =  os.getcwd() + '/images' + '/scheme.jpg'
scheme = base64.b64encode(open(scheme_filename, 'rb').read())
xgb_filename =  os.getcwd()+ '/images' + '/xgboost_algorithm.jpg'
xgb_img = base64.b64encode(open(xgb_filename, 'rb').read())

### Load dataframes to print
data_head_30 = pd.read_csv(pwd + "/data_head_30.csv", sep=",")
data_description = pd.read_csv(pwd + "/data_description.csv", sep=",")
data_description.rename( columns={'Unnamed: 0':''}, inplace=True)
unlabeled_data_with_results = pd.read_csv(pwd + "/unlabeled_data_with_results.csv", sep=",")

### Load data to process
data = pd.read_csv(pwd + "/dataset.txt", sep=",", index_col=[0])
data= data.dropna()
X = data.drop(["Class"], axis = 'columns')
y = data["Class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

### Pie plot
values_pie_train=pd.DataFrame(y_train).value_counts().tolist()
values_pie_test=pd.DataFrame(y_test).value_counts().tolist()
labels_pie=['Target value 0', 'Target value 1']

### Correlation plot
### Feature correlation ###
data_to_corr = X_train
corr_matrix = data_to_corr.corr()
corr_matrix_abs = data_to_corr.corr().abs()
### Select upper triangle of correlation matrix
upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(np.bool))

### Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >0.95)]

### Drop features 
data_correlation_95 = data_to_corr
data_correlation_95.drop(to_drop, axis=1, inplace=True)
print(data_correlation_95.shape)

### PCA plots
variance_explained = 0.9
features_to_PCA= data_correlation_95.columns.tolist()

pca, PCs_values = pca_func.PCA_calculation(data_correlation_95, features_to_PCA)
normalized_scores, feature_importance, selected_features, threshold= pca_func.PCA_feature_importance_selection(pca, features_to_PCA, variance_explained)
PCs, explained_variance_ratio, variance_exp_cumsum = pca_func.PCA_explained_variance(pca, features_to_PCA)

### Info about data 
labeled_cnt =data.shape[0]
unlabeled_cnt = unlabeled_data_with_results.shape[0]
numeric_features = X.shape[1]
numeric_features,labeled_cnt, unlabeled_cnt
gaussian_features = []
for i in X.columns:
    if shapiro(X[i])[1]>=0.05:
        gaussian_features.append(i)
gaussian_features_cnt = len(gaussian_features)  

### Load and evaluate the model
model_filename = "model.pkl"
xgb_model_loaded = pickle.load(open(model_filename, "rb"))
X_test = X_test[selected_features]
y_pred_xgb = xgb_model_loaded.predict(X_test)
y_proba_xgb = xgb_model_loaded.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred_xgb)
FP = confusion_matrix(y_test, y_pred_xgb)[0][1]
FN = confusion_matrix(y_test, y_pred_xgb)[1][0]
accuracy = round(100 - ((FP + FN)*100/len(y_pred_xgb)), 2)
auc = metrics.roc_auc_score(y_test, y_pred_xgb)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_xgb)

app = Dash(external_stylesheets=[dbc.themes.LUMEN])

app.layout = html.Div([html.H1(children='MNM Diagnostics - task solution'), html.H2(children='Description'),
                       html.Div(children='Create a classifier based on the provided small dataset. The target to predict is in the Class attribute and the positive class encoded as 1. Prepare a notebook demonstrating and justifying the steps you have taken to arrive at the final model. Furthermore, create an accompanying data visualization app (e.g., in Dash or Shiny) which will allow for the analysis of feature distributions and correlations between the features. If you are familiar with Docker, place the app in a Docker container.', style={'margin-bottom': '20px'}),
                       html.H2(children='Subset of data:'), 
                       html.Div(children=dash_table.DataTable(
                        data=data_head_30.to_dict('records'),
                        columns=[{'id': c, 'name': c} for c in data_head_30.columns],
                        fixed_rows={'headers': True},
                        style_cell={
                            'minWidth': 95, 'maxWidth': 95, 'width': 95
                        }),style={'margin-bottom': '20px'} ),
                       dcc.Markdown('''

                            ## After data preprocessing we know that, we have: 
                            * **Numeric features:** {}
                            * **Labeled samples:** {}
                            * **Unlabeled samples:** {}
                            * **Dataset is balanced**
                            * **Number of features with Gaussian distribution:** {} 

                        '''.format(numeric_features, labeled_cnt, unlabeled_cnt, gaussian_features_cnt)),
                        html.Div(children=[
                        html.H2(children='Target variable distribution - train and test datasets'),
                        html.Div(children=[
                        dcc.Graph(id='pie', style={'display': 'inline-block'}, 
                                figure = {'data': [go.Pie(labels=labels_pie, values=values_pie_train)]}),
                        dcc.Graph(id='pie2', style={'display': 'inline-block'}, 
                                figure = {'data': [go.Pie(labels=labels_pie, values=values_pie_test)]})])]),         

                        html.H2(children='Labelled data - descriptive statistics'),
                        html.Div(children=dash_table.DataTable(
                        data=data_description.to_dict('records'),
                        columns=[{'id': c, 'name': c} for c in data_description.columns],
                        fixed_rows={'headers': True},
                        style_cell={
                            'minWidth': 95, 'maxWidth': 95, 'width': 95
                        }),style={'margin-bottom': '20px'}),
                        html.Div([html.Img(src='data:image/png;base64,{}'.format(scheme.decode()), style={'height':'70%', 'width':'70%',  "display": "block", "margin-left": "auto","margin-right": "auto",})]), 
                        dcc.Graph(id='correlation', style={
                                                'height': 800,
                                                'width': 800,
                                                "display": "block",
                                                "margin-left": "auto",
                                                "margin-right": "auto",
                                                },
                        figure = {'data': [go.Heatmap(x = corr_matrix.columns,y = corr_matrix.index, z = np.array(corr_matrix))], 
                        'layout':go.Layout(height = 800, width =800, title='Correlation matrix')}), 
                        html.Div(children=[
                        dcc.Graph(id='feature importance', 
                                figure = {'data': [go.Bar(x=feature_importance,y=normalized_scores, orientation='v', name = 'feature importance'), go.Scatter(x=feature_importance, y = [threshold for i in range(len(feature_importance))], name='threshold', mode='lines', line_color='#ff7d76')], 
                                          'layout': go.Layout(height = 500, 
                                                             width =1600, 
                                                             title = 'Feature importance based on PCA analysis')}),
                        dcc.Graph(id='variance', 
                                figure = {'data': [go.Bar(x=PCs,y=explained_variance_ratio, orientation='v', name = 'Individual explained variance'), go.Scatter(x=PCs, y = variance_exp_cumsum, name = 'Cumulative explained variance',mode = 'lines+markers',line_width = 1)], 
                                'layout':go.Layout(height = 500, width =1600, title='PCA explained variance')})

                        ]), 
                        html.Div([html.H2("Model evaluation - metrics"),
                        html.Div([html.Img(src='data:image/png;base64,{}'.format(xgb_img.decode()), style={'height':'80%', 'width':'80%',  "display": "block", "margin-left": "auto","margin-right": "auto",})]), 
                        
                        html.Div(children=[dcc.Graph(id = 'ROC curve', style={'display': 'inline-block'}, figure = { 'data': [go.Scatter(x=fpr, y=tpr, 
                                                                        mode='lines', 
                                                                        line=dict(color='darkorange', width=2),
                                                                        name='ROC curve (area = %0.2f)' % auc), 
                                                                        go.Scatter(x=[0, 1], y=[0, 1], 
                                                                        mode='lines', 
                                                                        line=dict(color='navy', width=2, dash='dash'), name = 'Random guess', showlegend=True)], 
                                                                        'layout': go.Layout(height = 600, width =700, title='ROC (Receiver operating characteristic) plot', 
                                                                                            xaxis=dict(title='False Positive Rate'), yaxis=dict(title='True Positive Rate'))}), 
                                            dcc.Graph(id='confusion matrix', style={'display': 'inline-block'}, 
                                                        figure = {'data': [go.Heatmap(z = np.array(cm),
                                                        text=[['TP', 'FP'],
                                                                ['FN', 'TN']],
                                                        texttemplate="%{text}",
                                                        textfont={"size":20}, colorscale = 'Reds')], 
                                                                'layout':go.Layout(height = 500, width =500, title='Confusion matrix')})]),                   

                                            dcc.Markdown('''

                                                        * ** False positives (type I error): {} **
                                                        * ** False negatives (type II error): {} **
                                                        * ** Accuracy: {}% ** 

                                                    '''.format(FP, FN, accuracy))]), 
                        html.Div([html.H4("Unlabeled data - results:")]), 
                        dash_table.DataTable(
                        data=unlabeled_data_with_results.to_dict('records'),
                        columns=[{'id': c, 'name': c} for c in unlabeled_data_with_results.columns],
                        fixed_rows={'headers': True},
                        style_cell={
                            'minWidth': 95, 'maxWidth': 95, 'width': 95
                        }),

                        ])




server = app.server

if __name__ == '__main__':
    app.run()