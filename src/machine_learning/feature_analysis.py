import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

_bar_columns = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status','stroke']
_hist_columns = ['age', 'avg_glucose_level', 'bmi']

def plot_initial_graph(dataframe: pd.DataFrame):
  
  figBar, axes = plt.subplots(2,4, figsize=(15,10))

  colors=['blue','orange', 'green', 'yellow']

  col_idx = 0
  for i in range(2):
    for j in range(4):
      graph = axes[i][j]
      qtd = dataframe[_bar_columns[col_idx]].value_counts()
      x = np.arange(len(qtd))
      graph.bar(x, height=qtd.values, width=0.3, color=colors)
      graph.set_xticks(x, qtd.index)
      graph.set_title(_bar_columns[col_idx])

      for b in graph.containers:
        graph.bar_label(b)
      
      col_idx = col_idx + 1

  figHist, axesHist = plt.subplots(3,1, figsize=(15,10))
  
  for i in range(len(_hist_columns)):
    axesHist[i].hist(dataframe[_hist_columns[i]])
    axesHist[i].set_title(_hist_columns[i])

  figBar.tight_layout(pad=1.5, w_pad=2, h_pad=2)
  figHist.tight_layout(pad=1.5, w_pad=2, h_pad=2)
  plt.show()

def transform_feat(dataframe: pd.DataFrame) -> pd.DataFrame:
  new_gender = np.where(dataframe['gender']=='Male',1,0)
  dataframe['gender']=new_gender

  new_ever_marryed = np.where(dataframe['ever_married']=='Yes', 1, 0)
  dataframe['ever_married'] = new_ever_marryed

  multi_categ_columns = ['work_type', 'Residence_type', 'smoking_status']
  categ_processed = pd.get_dummies(dataframe[multi_categ_columns])
  dataframe = dataframe.join(categ_processed)

  for column in multi_categ_columns:
    del dataframe[column]
  
  return dataframe
  
def statistic_analysis(dataframe: pd.DataFrame):
  # transformed = transform_feat(dataframe)
  # print(transformed.describe())
  dataframe_stats = dataframe[_hist_columns].describe()
  print(dataframe_stats)

  figHist, axesHist = plt.subplots(3,1, figsize=(15,10))

  for i in range(len(_hist_columns)):
    m = dataframe_stats[_hist_columns[i]]['mean']
    std = dataframe_stats[_hist_columns[i]]['std']
    x = np.linspace(m - 3*std, m + 3*std, 100)
    p = norm.pdf(x, m, std)
    axesHist[i].plot(x, p, 'k')
    axesHist[i].hist(dataframe[_hist_columns[i]], density=True)
    # (n, bins, patches) = axesHist[i].hist(dataframe[_hist_columns[i]])
    # intervalss = bins.tolist()
    # stroke_by_interval =[]
    # for j in range(len(intervalss)-1):
    #   start = intervalss[j]
    #   end = intervalss[j+1]
    #   stroke_by_interval.append(dataframe.query(f'{_hist_columns[i]} >= {start} & {_hist_columns[i]} < {end}').count())

    # axesHist[i].hist(stroke_by_interval, bins=intervalss)
    axesHist[i].set_title(_hist_columns[i])

  figHist.tight_layout()
  plt.show()

  with_stroke = dataframe.query('stroke == 1')
  without_stroke = dataframe.query('stroke == 0')

  figVar, axesVar = plt.subplots(3,1, figsize=(15,10))

  for i in range(len(_hist_columns)):
    m1 = with_stroke[_hist_columns[i]].mean()
    std1 = with_stroke[_hist_columns[i]].std()
    x1 = np.linspace(m1 - 3*std1, m1 + 3*std1, 100)
    p1 = norm.pdf(x1, m1, std1)
    axesVar[i].plot(x1, p1, 'b')

    m2 = without_stroke[_hist_columns[i]].mean()
    std2 = without_stroke[_hist_columns[i]].std()
    x2 = np.linspace(m2 - 3*std2, m2 + 3*std2, 100)
    p2 = norm.pdf(x2, m2, std2)
    axesVar[i].plot(x2, p2, 'r')

    axesVar[i].set_title(_hist_columns[i])

    varA = (std1**2) / with_stroke[_hist_columns[i]].count()
    varB = (std2**2) / without_stroke[_hist_columns[i]].count()
    delta = np.sqrt( varA + varB )
    teste = (m1 - m2) / delta

    print(f'TESTE do campo {_hist_columns[i]} : {teste}')
  
  figVar.tight_layout()
  plt.show()

  transformed = transform_feat(dataframe)
  other_cols = [x for x in transformed.columns.to_list() if x not in _hist_columns and x!='stroke']
  transformed_stats = transformed[other_cols].describe()
  print(transformed_stats)

def oversample(dataframe: pd.DataFrame):
  classes = dataframe['stroke'].value_counts().to_dict()
  most = max(classes.values())
  classes_list = []
  for key in classes:
      classes_list.append(dataframe[dataframe['stroke'] == key]) 
  classes_sample = []
  for i in range(1,len(classes_list)):
      classes_sample.append(classes_list[i].sample(most, replace=True))
  df_maybe = pd.concat(classes_sample)
  final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
  final_df = final_df.reset_index(drop=True)
  return final_df

def start():
  dataframe = pd.read_csv("database/brain_stroke.csv")
  statistic_analysis(dataframe)
  print("Oversample")
  dataframe = oversample(dataframe)
  statistic_analysis(dataframe)
