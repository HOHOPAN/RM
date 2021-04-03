import time
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn import datasets
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

Orig_data=pd.read_csv('date.csv')
df_data=Orig_data.values

scaler = preprocessing.StandardScaler().fit(df_data)
a = scaler.transform(df_data)
xx = a[:,0:6]
yy = a[:,7]
XX, YY = shuffle(xx, yy, random_state=7)

num_training = int(0.8 * len(XX))
X_train, Y_train = XX[:num_training], YY[:num_training]
X_test, Y_test = XX[num_training:], YY[num_training:]

data=df_data[:,0:6]
target=df_data[:,11]
x, y = shuffle(data, target, random_state=7)

num_training = int(0.8 * len(x))

x_train, y_train = x[:num_training], y[:num_training]

x_test, y_test = x[num_training:], y[num_training:]

lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
param = {'num_leaves':31, 'num_trees':100, 'objective':'regression'}
t2 = time.time()
num_round = 10
bst = lgb.train(param, lgb_train, num_round, valid_sets=[lgb_eval])
lg_fit = time.time() - t2
print('lgb训练用时： ', lg_fit, '秒')


t3 = time.time()
KNN_regressor = KNeighborsRegressor()
KNN_regressor.fit(x_train, y_train)
knn_fit = time.time() - t3
print('KNN训练用时： ', knn_fit, '秒')



t4= time.time()
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(x_train, y_train)
ad_fit = time.time() - t4
print('ad训练用时： ', ad_fit, '秒')



t6 = time.time()
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
         'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(x_train, y_train)
gb_fit = time.time() - t6
print('gb训练用时： ', gb_fit, '秒')


# 使用带ANN算法的决策树模型进行拟合fit代表拟合
t16 = time.time()
model_mlp = MLPRegressor(hidden_layer_sizes=(60,60),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
                         random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                         early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model_mlp.fit(X_train, Y_train)
ANN_fit = time.time() - t16
print('ANN训练用时： ', ANN_fit, '秒')



t18 = time.time()
y_pred_ANN = model_mlp.predict(X_test)
print(Y_test)
print(y_pred_ANN)
ANN_predict = time.time() - t18
R2=r2_score(Y_test,y_pred_ANN)
evs = explained_variance_score(Y_test, y_pred_ANN)
print("###ANN学习效果###")
print('预测用时： ', t18, '秒')
print('ANN预测用时： ', ANN_predict, '秒')
print("R2: %.4f" % R2)
print(u"解释方差分=", round(evs, 2))



t10 = time.time()
y_pred_lgb = bst.predict(x_test, num_iteration=bst.best_iteration)
print(y_test)
print(y_pred_lgb)
lgb_predict = time.time() - t10
R2=r2_score(y_test,y_pred_lgb)
evs = explained_variance_score(y_test, y_pred_lgb)
print("###lgb学习效果###")
print('预测用时： ', t10, '秒')
print('lgb预测用时： ', lgb_predict, '秒')
print("R2: %.4f" % R2)
print(u"解释方差分=", round(evs, 2))


t11 = time.time()
y_pred_KNN = KNN_regressor.predict(x_test)
print(y_test)
print(y_pred_KNN)
KNN_predict = time.time() - t11
R2=r2_score(y_test,y_pred_KNN)
evs = explained_variance_score(y_test, y_pred_KNN)
print("###KNN学习效果###")
print('预测用时： ', t11, '秒')
print('KNN预测用时： ', KNN_predict, '秒')
print("R2: %.4f" % R2)
print(u"解释方差分=", round(evs, 2))

# 查看对AdaBoost进行改进之后的算法
t12 = time.time()
y_pred_ab = ab_regressor.predict(x_test)
print(y_test)
print(y_pred_ab)
R2=r2_score(y_test,y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
print('###AdaBoost###')
ad_predict = time.time() - t12
print('ad预测用时： ', ad_predict, '秒')
print("R2: %.4f" % R2)
print(u"解释方差分=", round(evs, 2))


t14 = time.time()
y_pred_gb = clf.predict(x_test)
print(y_test)
print(y_pred_gb)
R2=r2_score(y_test,y_pred_gb)
evs = explained_variance_score(y_test, y_pred_gb)
print('###gboost###')
gb_predict = time.time() - t14
print('gb预测用时： ', gb_predict, '秒')
print("R2: %.4f" % R2)
print(u"解释方差分=", round(evs, 2))

