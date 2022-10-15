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
  new_gender = np.where(dataframe['gender']=='Male',0,1)
  dataframe['gender']=new_gender

  new_ever_marryed = np.where(dataframe['ever_married']=='Yes', 1, 0)
  dataframe['ever_married'] = new_ever_marryed

  multi_categ_columns = ['work_type', 'Residence_type', 'smoking_status']
  categ_processed = pd.get_dummies(dataframe[multi_categ_columns])
  print(categ_processed.head())
  dataframe = dataframe.join(categ_processed)

  for column in multi_categ_columns:
    del dataframe[column]
  
  return dataframe
  
def statistic_analysis(dataframe: pd.DataFrame):
  # transformed = transform_feat(dataframe)
  # print(transformed.describe())
  dataframe_stats = dataframe.describe()
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
  


def start():
  dataframe = pd.read_csv("database/brain_stroke.csv")
  # plot_graph(dataframe)
  statistic_analysis(dataframe)