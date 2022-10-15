from __future__ import annotations
from itertools import count
import os
import pickle
from unicodedata import category
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.neural_network import MLPClassifier
from dtreeviz.trees import *

def missing_data(dataframe: pd.DataFrame) -> pd.DataFrame:
  pass


def pre_processing(dataframe: pd.DataFrame) -> pd.DataFrame:

  new_gender = np.where(dataframe['gender']=='Male',0,1)
  dataframe['gender']=new_gender

  new_ever_marryed = np.where(dataframe['ever_married']=='Yes', 1, 0)
  dataframe['ever_married'] = new_ever_marryed

  categ_columns = ['work_type', 'Residence_type', 'smoking_status']
  categ_processed = pd.get_dummies(dataframe[categ_columns])
  print(categ_processed.head())
  dataframe = dataframe.join(categ_processed)
  
  # prep_encoder = OneHotEncoder(handle_unknown='ignore') #OneHotEncoder(categories= 'auto')
  # transformer = make_column_transformer(
  #   (prep_encoder, categ_columns),
  #   remainder='passthrough'
  # )
  # transformed = transformer.fit_transform(dataframe[categ_columns])

  for column in categ_columns:
    del dataframe[column]
  
  # transformed_df = pd.DataFrame(transformed, columns= transformer.get_feature_names_out())
  # dataframe = dataframe.join(transformed_df)

  new_stroke = pd.Categorical(dataframe['stroke'])
  new_stroke = new_stroke.rename_categories({0: 'Non-stroke',1:'Stroke'})
  dataframe['stroke_categ'] = new_stroke

  return dataframe

def data_graph(dataframe: pd.DataFrame):
  bar_columns = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status','stroke']
  hist_columns = ['age', 'avg_glucose_level', 'bmi']
  
  figBar, axes = plt.subplots(2,4, figsize=(15,10))

  colors=['blue','orange', 'green', 'yellow']

  col_idx = 0
  for i in range(2):
    for j in range(4):
      graph = axes[i][j]
      qtd = dataframe[bar_columns[col_idx]].value_counts()
      x = np.arange(len(qtd))
      hbar = graph.bar(x, height=qtd.values, width=0.3, color=colors)
      graph.set_xticks(x, qtd.index)
      graph.set_title(bar_columns[col_idx])
      # axes[i][j].bar_label(hbar, padding=2)

      for b in graph.containers:
        graph.bar_label(b)
      
      # for k, k_qtd in enumerate(qtd.tolist()):
      #   axes[i][j].text(k + .25, qtd + 3, str(k_qtd))

      col_idx = col_idx + 1
  
  # fig1, ax1 = plt.subplots()
  # ax1.text

  # qtd = dataframe['stroke'].value_counts()
  # y_position = np.arange(len(qtd))
  
  # hbar = ax1.bar(y_position, height=qtd.values, width=0.7, color= cm)
  # ax1.set_xticks(y_position, qtd.index)
  # ax1.set_title('Quantidade de pessoas com AVC?')

  # ax1.bar_label(hbar, padding=2)


  figHist, axesHist = plt.subplots(3,1, figsize=(15,10))
  
  for i in range(len(hist_columns)):
    axesHist[i].hist(dataframe[hist_columns[i]])
    axesHist[i].set_title(hist_columns[i])


  figBar.tight_layout(pad=1.5, w_pad=2, h_pad=2)
  figHist.tight_layout(pad=1.5, w_pad=2, h_pad=2)
  plt.show()


def trainning():

  dataframe = pd.read_csv("database/brain_stroke.csv")

  # print(f'Dimensao: { brain_stroke.shape}')
  # print(f'Campos: { list(brain_stroke.keys())}')
  # print(f'Tipo de dados: {brain_stroke.dtypes}')
  # print(brain_stroke.describe())

  result = dataframe.query('age < 18 and stroke==1')
  print(result)

  data_graph(dataframe)

  dataframe = pre_processing(dataframe)
  print(dataframe.dtypes)


  classes = ['Non-stroke', 'Stroke']
  
  features = dataframe.columns.values.tolist()
  features = [c for c in features if c.startswith('stroke') == False]
  X = dataframe[features]
  y = dataframe['stroke']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

  print("Arvore de descisão")
  decision_tree_model = DecisionTreeClassifier(random_state=0, criterion='entropy', class_weight='balanced')
  decision_tree_model = decision_tree_model.fit(X_train, y_train)
  
  metrics_name = ['accuracy', 'precision_macro', 'recall_macro']
  metrics = cross_validate(decision_tree_model, X_train, y_train, cv=5, scoring=metrics_name)
  for met in metrics:
    print(f"- {met}:")
    print(f"-- {metrics[met]}")
    print(f"-- {np.mean(metrics[met])} +- {np.std(metrics[met])}\n") 
  
  tree_predicts = cross_val_predict(decision_tree_model, X, y, cv=5)
  print(tree_predicts)


  y_prediction = decision_tree_model.predict(X_test)
  print("Acurácia de previsão:", accuracy_score(y_test, y_prediction))
  print(classification_report(y_test, y_prediction, target_names=classes))
  # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


  # viz = dtreeviz(decision_tree_model,
  #             X_train,
  #             y_train,
  #             target_name="stroke_categ",
  #             feature_names=features,
  #             class_names=classes)  

  # viz.view()

  print("Rede neural")
  neural_network_model = MLPClassifier()
  neural_network_model = neural_network_model.fit(X_train, y_train)
  
  neural_pred = neural_network_model.predict(X_test)
  print("Acurácia de previsão:", accuracy_score(y_test, neural_pred))
  print(classification_report(y_test, neural_pred, target_names=classes))
  