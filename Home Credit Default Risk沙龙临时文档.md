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
- 4min ：Featuretools
- 4min ：自编码器
- 2min ：-----**讨论**-----

#### 5. “剑走偏锋”还是“旁门左道”（10min）
- 4min ：主流高分方案： **stacking** & **blending**
- 6min ：“**剑走偏锋**”高分方案

#### 6. 讨论

# 一、Kaggle是个啥？

![kaggle介绍](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/image-20200411224433386.png)

### 1.1 Kaggle：

​	有着丰富的**数据挖掘**比赛题目，大量公开的**代码方案**

### 1.2 今天的topic：**Home Credit Default Risk**

#### (1)  业务背景（目标是什么）

​		希望能通过数据挖掘和机器学习算法来估计客户的贷款违约概率PD

#### (2)  数据（哪些信息摸得到）

  - **客户申请表** application_train/test 
  - **客户信用记录历史**(月数据) bureau/bureau_balance
  - **客户账户余额&贷款历史**(月数据) POS_CASH_balance 
  - **客户信用卡信息表** (月数据)credit_card_balance
  - **客户历史申请表**previous_application
  - **客户先前信用卡还款记录** installments_payments


![数据表关系介绍](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/home%20credit%20%E6%95%B0%E6%8D%AE%E8%A1%A8%E5%85%B3%E7%B3%BB.png)


#### (3)  结果评判标准

​		测试数据集上的**AUC**    

- - - - - - - - - - - - - - - - - - - - - - - - - - 

# 二、快速上手：A simple & quick solution

我们来看看：一个最简单的**逻辑回归**建模，要经历哪些**典型步骤**

### 2.1 利用pandas.read_csv()读取数据
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

### 2.2 数据处理

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
### 2.3 训练模型：逻辑回归  
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

- - - - - - - - - - - - - - - - - - - - - - - - - - 

# 三、从逻辑回归 → XGBoost 

### 3.1 逻辑回归Logistic Regression

![逻辑回归公式](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%9A%84%E5%85%AC%E5%BC%8F-%E5%9B%9E%E5%88%B0sklearn.png)

大都是：基于**一阶梯度、二阶梯度**迭代寻找w, b；

回到`sklearn.linear_model.LogisticRegression()`的参数上：

> **solver**：优化算法选择
>
> - *liblinear*：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
> - *lbfgs*：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
> - *newton-cg*：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。只用于L2
> - *sag*：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。只用于L2
> - *saga*：线性收敛的随机优化算法的的变重。只用于L2
>
> **penalty**：**惩罚项**
> 	str类型，默认为l2。newton-cg、sag和lbfgs求解算法只支持L2规范,L2假设的模型参数满足高斯分布。
> 	l1:L1G规范假设的是模型的参数满足拉普拉斯分布.

> **tol**：**停止求解的标准**，float类型，默认为1e-4。就是求解到多少的时候，停止，认为已经求出最优解。

> **c**：**正则化系数λ的倒数**，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。

> **fit_intercept**：是否存在截距或偏差，bool类型，默认为True。

> ==**class_ weight**==：
> ​	用于标示分类模型中**各种类型的权重**，可以是一个字典或者’balanced’字符串，默认为不输入，也就是不考	虑权重，即为None。
> ​	如果选择输入的话，可以选择balanced让类库自己计算类型权重，或者自己输入各个类型的权重。
>
> **random_state**：随机数种子，int类型，可选参数，默认为无，仅在正则化优化算法为sag,liblinear时有用。
>
> **max_iter**：算法收敛最大迭代次数，int类型，默认为10。仅在正则化优化算法为newton-cg, sag和lbfgs才	有用，算法收敛的最大迭代次数。

> **multi_class**：分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。ovr即前面提到的	one-vs-rest(OvR)，而multinomial即前面提到的many-vs-many(MvM)，区别主要在**<u>多元逻辑回归</u>**上。

### 3.2 Boosting策略

我们总是能听到Catboost、XGBoost、LightGBM、GBDT这类：

以**决策树**模型为基学习器，还带着个**Boost**为形容词的高大上算法，我们快速了解一下吧！

#### (1) CART (Classification & Regression Tree)：一个能回归也能分类的树模型

**CART**作为Boosting策略使用最广泛的基学习器：

- 用于**回归** *regression*：最小化MSE（均方误差）
- 用于**分类** *classification*：最小化GINI系数

#### (2) 作用在梯度上的Boosting （Gradient Boosting）

**后一个**基学习器   → ***拟合*** →   **前一个**基学习器的**残差**

`这里的残差在工程上一般定义为：损失函数的负梯度`

![XGB梯度提升示意图](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/XGB%E6%8F%90%E5%8D%87%E7%A4%BA%E6%84%8F%E5%9B%BE.png)

#### （3）瞅一瞅XGBoost

![XGboost公式](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/XGBoost%E5%85%AC%E5%BC%8F.png)

2. 部分参数一览

   ```python
   import xgboost as xgb
   xgb.XGBRegressor({'n_estimators': 500,'eta': 0.3,  'gamma': 0, 'max_depth': 6,\
                     'reg_lambda': 1, 'reg_alpha': 0,'seed': 33})
   #采样、叶子节点细节的相关参数就暂时忽略吧：
   #'min_child_weight': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1
   ```

> **n_estimators**：弱学习器的数量
>
> **max_depth**：默认是6，树的最大深度，值越大，越容易过拟合；[0，∞]
>
> **eta** ： 默认是0.3，别名是 **leanring_rate**，更新过程中用到的收缩步长，在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守；[0,1]
>
> **seed**：随机数种子，相同的种子可以复现随机结果，用于调参！
>
> ----以下是**正则项**参数-----
>
> **gamma**：默认是0，别名是 min_split_loss，gamma值越大，算法越保守（越不容易过拟合）；[0，∞]
>
> **lambda**：默认是1，别名是reg_lambda，L2 正则化项的权重系数，越大模型越保守；
>
> **alpha**：默认是0，别名是reg_alpha，L1 正则化项的权重系数，越大模型越保守；

- - - - - - - - - - - - - - - - - - - - - - - - - - 
# 四、特征工程与Featuretools

这是一个一时半会说不完的门类，简单来说：

- 分析数据、处理数据
- 衍生变量、无中生有
- 筛选变量、万里挑一  

### 4.1 一个很厉害的工具：Featuretools

```python
! pip install featuretools
import featuretools as ft
#创建实体
es = ft.EntitySet(id = 'clients')
#添加clients实体
es = es.entity_from_dataframe(entity_id = 'clients', dataframe = clients, 
                              index = 'client_id', time_index = 'joined')
#聚合特征，通过指定聚合agg_primitives和转换trans_primitives生成新特征
eatures, feature_names = ft.dfs(entityset = es, target_entity = 'clients', agg_primitives = ['mean', 'max', 'percent_true', 'last'], trans_primitives = ['subtract', 'divide'])
#聚合agg：sum、mean
#转换trans：相加、相减、相除等
```

### 4.2 试试自编码器

1. **线性**自编码器：PCA
2. **非线性**自编码器：神经网络、树模型等

![自编码器](https://github.com/RainFlanker/rainflanker.github.io/blob/master/images/%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.png)

- - - - - - - - - - - - - - - - - - - - - - - - - - 

# 五、“剑走偏锋”还是旁门左道

### 5.1 主流高分操作：

- **Stacking**

  ![Stacking, Blending and Stacked Generalization](https://www.chioka.in/wp-content/uploads/2013/09/stacking.png)

- **Bagging**

  Bootstrap Aggregating的缩写，采用一种自助采样的方法（boostrap sampling）每次从数据集中随机选择一个subset，然后放回初始数据集，下次取时，该样本仍然有一定概率取到。然后根据对每个subset训练出一个基学习器，然后将这些基学习器进行结合（算术、几何平均等）

### 5.2 百花齐放之“剑走偏锋”：

- **用户画像特征：**

  第5名 (Kraków, Lublin i Zhabinka)

  原贴: [Overview of the 5th solution +0.004 to CV/Private LB of your model](https://link.zhihu.com/?target=https%3A//www.kaggle.com/c/home-credit-default-risk/discussion/64625)

  - 神经网络构建用户画像，加入stacking

  来自不同表格的信息交互很难通过人为判断进行提取, 因此尝试用NN构建用户画像, 将其转化为用户分类问题. 在每张数据表上(application表除外)都可以构建一个用户的向量(每个月的数据), 然后将这些向量合并到一起, 得到一个较为稀疏的用户画像. 然后构建如下NN: (1) 对每个向量进行normalization: 除以最大值; (2) 输入为96个向量, 每个月为一个向量; (3) 1维卷积层(Conv1D); (4) Bidirectional LSTM; (5) Dense层; (6) 输出.

- **挖掘利率信息**(还是上面那个第5名)

   关键点是AMT*ANNUITY 包含了利率， 基于 AMT*CREDIT, AMT*ANNUITY, and CNT*PAYMENT 我们可以衍生出利率。

```python
prev_app['INTEREST'] = prev_app['CNT_PAYMENT']*prev_app['AMT_ANNUITY'] \
												-prev_app['AMT_CREDIT']
prev_app['INTEREST_RATE'] = 2*12*prev_app['INTEREST']/(prev_app['AMT_CREDIT']*\
                                                       (prev_app['CNT_PAYMENT']+1))
prev_app['INTEREST_SHARE'] = prev_app['INTEREST']/prev_app['AMT_CREDIT']
```

- **Pseudo Data Augmentation**破解违约定义(一个日本小哥)

   第27名 (nyanp)

  原贴: [Pseudo Data Augmentation (27th place short writeup)](https://link.zhihu.com/?target=https%3A//www.kaggle.com/c/home-credit-default-risk/discussion/64693)

  这哥们有点意思：

  训练模型去找**违约定义**里面的未知参数**X、Y**，然后通过把没有打标的**previous历史**客户记录全**打上标**，样本就变多了。

​	 ***打标为1：***

```python
客户存在还款困难，定义如下：
he/she had late payment more than **X** days on at least one of the first **Y** installments of the loan in our sample,
```

​	***打标为0：***

```
所有其他情况
```

- - - - - - - - - - - - - - - - - - - - - - - - - - 

# 六、尽情讨论
