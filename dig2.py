import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
file_path = 'happiness_train_abbr.csv'

# Read Excel file specifying the engine
data = pd.read_csv('happiness_train_abbr.csv', parse_dates=['survey_time'], encoding='utf-8')
test = pd.read_csv("happiness_test_abbr.csv", parse_dates=["survey_time"], encoding='latin-1')
# 查看数据的前几行


# 处理缺失值（例如，用均值填充）
data.fillna(0, inplace=True)
data = data.loc[data['happiness'] != -8]
# 分离特征和标签

data['survey_time'] = data['survey_time'].dt.year
test['survey_time'] = test['survey_time'].dt.year
data['Age'] = data['survey_time']-data['birth']
test['Age'] = test['survey_time']-test['birth']
del_list=['survey_time','birth']
figure,ax = plt.subplots(1,1)
data['Age'].plot.hist(ax=ax,color='blue')
combine=[data,test]
for dataset in combine:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 80), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 80, 'Age'] = 5
figure1,ax1 = plt.subplots(1,5,figsize=(18,4))
data['happiness'][data['Age']==1].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1[0],shadow=True)
data['happiness'][data['Age']==2].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1[1],shadow=True)
data['happiness'][data['Age']==3].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1[2],shadow=True)
data['happiness'][data['Age']==4].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1[3],shadow=True)
data['happiness'][data['Age']==5].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1[4],shadow=True)
