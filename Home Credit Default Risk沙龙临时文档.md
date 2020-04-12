# Home Credit Default Risk 沙龙分享2020-0414  
- - - - - - - - - - - - - - - - - - - - - - - - - - 

#  流程速览：
#### 1. 逛一下Kaggle网站  （10min）
- 2min : 逛主界面
- 4min  : 浏览今天题目：Home credit default risk
- 4min : -----**讨论**-----

#### 2. A quick simple solution （10min）
- 3min : 看代码
- 5min：主持人统一过一遍
- 2min ：-----**讨论**-----
  
#### 3. 从逻辑回归→回归树→boost家族（10min）
- 2min ：逻辑回归的 ***极大似然估计***
- 2min ：用决策树来处理***似然函数***
- 4min ：基于加法模型的Boost系列
- 2min ：-----**讨论**-----

#### 4. 特征工程与featuretools （10min）
- 2min ：Featuretools
- 3min ：自编码器
- 2min ：“业务含义”造变量
- 3min ：-----**讨论**-----

#### 5. “剑走偏锋”还是“旁门左道”（10min）
- 4min ：主流高分方案： **stacking** & **blending**
- 6min ：“**剑走偏锋**”高分方案

#### 6. 讨论

# 一、Kaggle是个啥？

![kaggle介绍](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/image-20200411224433386.png)

### 1.1 Kaggle：

​	有着丰富的**数据挖掘**比赛题目，大量公开的**代码方案**

### 1.2 今天的topic：Home Credit Default Risk

#### (1)  业务背景（目标是什么）

​		希望能通过数据挖掘和机器学习算法来估计客户的贷款违约概率PD

#### (2)  数据（哪些信息摸得到）

  - **客户申请表** application_train/test 
  - **客户信用记录历史**(月数据) bureau/bureau_balance
  - **客户账户余额&贷款历史**(月数据) POS_CASH_balance 
  - **客户信用卡信息表** (月数据)credit_card_balance
  - **客户历史申请表**previous_application
  - **客户先前信用卡还款记录** installments_payments


![数据表关系介绍](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/image-20200411233751953.png)


#### (3)  结果评判标准

​		测试数据集上的**AUC**

# 三、快速上手：A simple & quick solution

我们来看看：一个最简单的**逻辑回归**建模，要经历哪些**典型步骤**

### 3.1 利用pandas.read_csv()读取数据
```python
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
pos_cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')
previous_app = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
```

### 3.2 数据处理

#### (1) 衍生变量：统计客户的历史贷款次数
```python
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU']\
  .count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
```

| 客户编号 SK_ID_CURR | Previous_loan_counts |
| :-----------------: | :------------------: |
|       1000001       |          7           |
|       1000002       |          8           |
|       1000003       |          4           |

#### (2) 继续对多张表按照客户编号SK_ID_CURR进行groupby

​	每个客户对应一行特征，这样依据*客户编号sk_id_curr*做横向merge简单好理解。
```python
# Grouping data  so  that we can merge all the files in 1 dataset
data_bureau_agg=bureau.groupby(by='SK_ID_CURR').mean()
data_credit_card_balance_agg=credit_card_balance.groupby(by='SK_ID_CURR').mean()
data_previous_application_agg=previous_app.groupby(by='SK_ID_CURR').mean()
data_installments_payments_agg=installments_payments.groupby(by='SK_ID_CURR').mean()
data_POS_CASH_balance_agg=pos_cash_balance.groupby(by='SK_ID_CURR').mean()

data_bureau_agg.head()
```

![data_bureau_agg](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/image-20200412001603264.png)


#### (3) 做left join
```python
def merge(df):
    df = df.join(data_bureau_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 
    df = df.join(bureau_counts, on = 'SK_ID_CURR', how = 'left')
    df = df.join(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
    df = df.join(data_credit_card_balance_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2')    
    df = df.join(data_previous_application_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2')   #这里的suffix是当两张表出现同样的列名时，对left table、right table增加不同的后缀
    df = df.join(data_installments_payments_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 
    return df

train = merge(app_train)
test = merge(app_test)
display(train.head())
```

![train.head](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/image-20200412002545469.png)

#### (4) 变量取值预处理

​	负的天数调整为**正数**，缺失值**fillna()**

```python 
 # Now we will convert days employed and days registration and days id publish to a positive no. 
def correct_birth(df): #负号并转换成年份   
    df['DAYS_BIRTH'] = round((df['DAYS_BIRTH'] * (-1))/365)
    return df
def convert_abs(df): #取绝对值
    df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
    df['DAYS_REGISTRATION'] = abs(df['DAYS_REGISTRATION'])
    df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])
    df['DAYS_LAST_PHONE_CHANGE'] = abs(df['DAYS_LAST_PHONE_CHANGE'])
    return df
 # Now we will fill misisng values in OWN_CAR_AGE. 
 #Most probably there will be missing values if the person does not own a car. So we will fill with 0
def missing(df):  #填充缺失值
    features = ['previous_loan_counts','NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_MEDI','OWN_CAR_AGE']
    for f in features:
        df[f] = df[f].fillna(0 )
    return df
def transform_app(df):
    df = correct_birth(df)
    df = convert_abs(df)
    df = missing(df)
    return df
all_data = transform_app(all_data)
```

数值变量进行值域放缩**MinMaxScaler()**
```python
from sklearn.preprocessing import MinMaxScaler
def encoder(df):
    scaler = MinMaxScaler()
    numerical = all_data.select_dtypes(exclude = ["object"]).columns
    features_transform = pd.DataFrame(data= df)
    features_transform[numerical] = scaler.fit_transform(df[numerical])
    display(features_transform.head(n = 5))
    return df
all_data = encoder(all_data)
```

类别型变量进行**LabelEncoder()**、**get_dummies()**
```python
from sklearn.preprocessing import LabelEncoder
all_data[col] = LabelEncoder().fit_transform(all_data[col])
all_data = pd.get_dummies(all_data)
```
### 3.3 训练模型：逻辑回归  
逻辑回归：LogisticRegression()

```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0, class_weight='balanced', C=100)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict_proba(X_test)[:,1]
```
这是一个验证集AUC=0.75的方案，过程比较基础，  
请问：    
`LogisticRegression(random_state=0, class_weight='balanced', C=100)`里面的参数设置起到什么作用呢？

# 四、从逻辑回归 → XGBoost

