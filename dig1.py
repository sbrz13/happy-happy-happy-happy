import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read Excel file specifying the engine
data = pd.read_csv('happiness_train_abbr.csv', parse_dates=['survey_time'], encoding='utf-8')
test = pd.read_csv("happiness_test_abbr.csv", parse_dates=["survey_time"], encoding='latin-1')
# 查看数据的前几行


# 处理缺失值（例如，用均值填充）
data.fillna(0, inplace=True)
data = data.loc[data['happiness'] != -8]
# 分离特征和标签


data1 = data[['happiness', 'edu', 'status_peer']]

plt.subplot(1, 2, 1)  # 1行2列的第1个
sns.violinplot(data=data1,x='happiness',y="edu")
plt.title('edu by Happiness Levels')
plt.xlabel('Happiness Levels')
plt.ylabel('edu')

# 创建收入的箱形图
plt.subplot(1, 2, 2)  # 1行2列的第2个
sns.violinplot(data=data1,x='happiness',y="status_peer")
plt.title('status_peer Comparison by Happiness Levels')
plt.xlabel('Happiness Levels')
plt.ylabel('status_peer')

# 显示图表
plt.tight_layout()
plt.show()



data2 = data[['happiness', 'equilty', 'marital']]

plt.subplot(1, 2, 1)  # 1行2列的第1个
sns.violinplot(data=data2,x='happiness',y="equilty")
plt.title('equilty by Happiness Levels')
plt.xlabel('Happiness Levels')
plt.ylabel('equilty')

# 创建收入的箱形图
plt.subplot(1, 2, 2)  # 1行2列的第2个
sns.violinplot(data=data1,x='happiness',y="marital")
plt.title('marital Comparison by Happiness Levels')
plt.xlabel('Happiness Levels')
plt.ylabel('marital')

# 显示图表
plt.tight_layout()
plt.show()