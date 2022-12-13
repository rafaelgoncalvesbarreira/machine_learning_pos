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
from sklearn.naive_bayes import GaussianNB
from dtreeviz.trees import *
import feature_analysis
from feature_analysis import oversample

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




def trainning():

  dataframe = pd.read_csv("database/brain_stroke.csv")

  dataframe = pre_processing(dataframe)
  dataframe = oversample(dataframe)

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

  conf_matrix = confusion_matrix(y_test, y_prediction)
  confussion_table = pd.DataFrame(data=conf_matrix, index=classes, columns=[x + "(prev)" for x in classes])
  print(confussion_table)

  print("\n########################\n\n\n\n")

  print("Rede neural")
  neural_network_model = MLPClassifier()
  neural_network_model = neural_network_model.fit(X_train, y_train)
  
  neural_pred = neural_network_model.predict(X_test)
  print("Acurácia de previsão redes neurais:", accuracy_score(y_test, neural_pred))
  print(classification_report(y_test, neural_pred, target_names=classes))

  conf_matrix = confusion_matrix(y_test, neural_pred)
  confussion_table = pd.DataFrame(data=conf_matrix, index=classes, columns=[x + "(prev)" for x in classes])
  print(confussion_table)

  #Gaussian Naïve-Bayes

  print("\n########################\n\n\n\n")
  print("Gauss Naive-Bayes")
  gnb = GaussianNB()
  gnb_model = gnb.fit(X_train, y_train)
  gnb_pred= gnb_model.predict(X_test)
  print("Acurácia de previsão naive-bayes:", accuracy_score(y_test, gnb_pred))
  print(classification_report(y_test, gnb_pred, target_names=classes))
  conf_matrix = confusion_matrix(y_test, gnb_pred)
  confussion_table = pd.DataFrame(data=conf_matrix, index=classes, columns=[x + "(prev)" for x in classes])
  print(confussion_table)
  