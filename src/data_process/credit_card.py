from .base_data_processor import BaseDataPreprocessor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class CreditCardDataPreprocessor(BaseDataPreprocessor):
    def __init__(self, config, model=None):
        super().__init__(config)
        
        # 定义数值型和分类型特征
        numeric_features = ['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                          'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                          'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
        
        # 根据model的normalize属性决定是否进行预处理
        normalize = model.normalize
        
        if normalize:
            transformers = [
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        else:
            all_features = numeric_features + categorical_features
            transformers = [('passthrough', 'passthrough', all_features)]
        
        # 创建预处理器
        preprocessor = ColumnTransformer(transformers=transformers)
            
        # 添加到预处理器列表
        self.add_preprocessor(preprocessor)
        
    def split_features_target(self, df):
        X = df.drop(['ID', 'default payment next month'], axis=1)
        y = df['default payment next month']
        return X, y
    
    def clean_data(self, X):
        # 性别映射 (1->0, 2->1)
        gender_mapping = {1: 0, 2: 1}
        X = X.copy()  
        X['SEX'] = X['SEX'].map(gender_mapping)
        
        # 教育程度重新映射
        X = X.assign(EDUCATION=X['EDUCATION'].replace([1,3,4,5,6], [3,1,0,0,0]))
        
        # 婚姻状态映射
        X = X.assign(MARRIAGE=X['MARRIAGE'].replace(3, 0))
        
        # 还款状况映射
        for i in range(1, 7):
            X = X.assign(**{f'PAY_{i}': X[f'PAY_{i}'].replace([-1, -2], [0, 0])})
        
        # 处理缺失值
        X = X.dropna()
        return X