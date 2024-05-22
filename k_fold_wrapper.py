import imp
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.metrics import mean_squared_error,r2_score
from function import adjust_sample
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor




class KFoldWapper(object):
    def __init__(self,layer_id,index,config,random_state):
        self.config=config
        self.name="layer_{}, estimstor_{}, {}".format(layer_id,index,self.config["type"])
        if random_state is not None:
            self.random_state=(random_state+hash(self.name))%1000000007
        else:
            self.random_state=None
        self.n_fold=self.config["n_fold"]
        self.estimators=[None for i in range(self.config["n_fold"])]
        self.config.pop("n_fold")
        self.estimator_class=globals()[self.config["type"]]
        self.config.pop("type")
    
    def _init_estimator(self):
        estimator_args=self.config
        est_args=estimator_args.copy()
        est_args["random_state"]=self.random_state
        return self.estimator_class(**est_args)
    
    def fit(self,x,y,y_valid,error_threshold,resampling_rate):
        kf=RepeatedKFold(n_splits=self.n_fold,n_repeats=1,random_state=self.random_state)

        # 困扰我的问题在weiping这里的实现方式是：
        # 先将所有的训练样本分为了k折，每次拿k-1折作为训练样本
        # 然后在每次对简单样本困难样本采样的时候，都是在这k-1折的训练样本中进行的
        # 然后每次的增强向量都是在OOF折外样本中预测的

        # 而我想的是分折前采样，这样就没有办法得到OOF样本的增强向量
        cv=[(t,v) for (t,v) in kf.split(x)]
        y_train_pred=np.zeros((x.shape[0],))
        for k in range(len(self.estimators)):
            est=self._init_estimator()
            train_id, val_id=cv[k]
            x_train,y_train=x[train_id],y[train_id]
            if y_valid is not None:
                x_train,y_train=adjust_sample(x_train,y_train,y_valid[train_id],error_threshold,resampling_rate)
            est.fit(x_train,y_train)
            y_pred=est.predict(x[val_id])
            y_train_pred[val_id]+=y_pred
            self.estimators[k]=est
        return y_train_pred

    def predict(self,x):
        pre_value=0
        for est in self.estimators:
            pre_value+=est.predict(x)
        pre_value/=len(self.estimators)
        return pre_value