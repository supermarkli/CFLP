import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
import logging
from data_process.credit_card import CreditCardDataPreprocessor
from utils.metrics import ModelMetrics
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
import copy

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.dropout1 = nn.Dropout(0.4)
        
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.dropout2 = nn.Dropout(0.3)
        
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.dropout3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.dropout1(x1)
        
        x2 = self.layer2(x1)
        x2 = self.dropout2(x2)
        x2 = x2 + x1[:, :64]
        
        x3 = self.layer3(x2)
        x3 = self.dropout3(x3)
        x3 = x3 + x2[:, :32]
        
        return self.output(x3)

class NeuralNetModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = None
        self.epochs = 100
        self.batch_size = 128
        self.learning_rate = 0.0005
        self.early_stopping_rounds = 500
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.normalize = True
        self.preprocessor = CreditCardDataPreprocessor(config=config, model=self)

        self.name = "NeuralNet"
        self.metrics = ModelMetrics()
        self.best_threshold = 0.5

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型
        
        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
            X_val: 验证数据特征，可选
            y_val: 验证数据标签，可选
        """
        logging.info("\n=== Training Start ===")
        logging.info(f"Training data size: {X_train.shape}")
        
        # 初始化网络(如果还未初始化)
        if self.network is None:
            input_dim = X_train.shape[1]
            self.network = NeuralNetwork(input_dim=input_dim).to(self.device)
            logging.info(f"Initialized neural network with input dimension: {input_dim}")
        
        # 保存验证集数据
        if X_val is not None and y_val is not None:
            self.X_val = X_val
            self.y_val = y_val
        
        # 打印类别分布
        pos_count = sum(y_train == 1)
        neg_count = sum(y_train == 0)
        logging.info("Class distribution in training set:")
        logging.info(f"  Positive samples: {pos_count}")
        logging.info(f"  Negative samples: {neg_count}")
        
        # 转换为tensor并移动到正确的device
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        pos_weight = torch.tensor(neg_count / pos_count).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化最佳验证指标
        best_val_loss = float('inf')
        best_val_f1 = 0
        no_improvement = 0
        
        # 训练循环
        for epoch in range(self.epochs):
            self.network.train()
            total_loss = 0
            
            # 批次训练
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.network(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # 计算平均损失
            avg_loss = total_loss / len(train_loader)
            
            # 在验证集上评估
            self.network.eval()
            with torch.no_grad():
                val_loss = 0
                val_f1 = 0
                if hasattr(self, 'X_val') and hasattr(self, 'y_val'):
                    val_outputs = self.network(torch.FloatTensor(self.X_val).to(self.device))
                    val_loss = criterion(val_outputs.squeeze(), 
                                      torch.FloatTensor(self.y_val.values).to(self.device)).item()
                    val_preds = (torch.sigmoid(val_outputs.squeeze()) > 0.5).float()
                    val_f1 = f1_score(self.y_val, val_preds.cpu().numpy())
                    
                    # 打印训练进度
                    logging.info(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
                    
                    # 检查是否需要早停
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_val_loss = val_loss
                        no_improvement = 0
                        # 保存最佳模型
                        self.best_model_state = copy.deepcopy(self.network.state_dict())
                    else:
                        no_improvement += 1
                        
                    if no_improvement >= self.early_stopping_rounds:
                        logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                        break
                else:
                    # 如果没有验证集,只打印训练损失
                    logging.info(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')
        
        # 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.network.load_state_dict(self.best_model_state)
        
        logging.info("Model training completed")

    def find_best_threshold(self, X_val, y_val):
        """找到最优的预测阈值"""
        self.network.eval()
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_val_tensor)
            y_pred_proba = torch.sigmoid(outputs).cpu().numpy()
        
        thresholds = np.arange(0.1, 0.9, 0.02)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        logging.info(f"Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

    def evaluate_loss(self, X_val, y_val, pos_weight=None):
        """评估验证集上的损失"""
        self.network.eval()
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_val_tensor)
            if pos_weight is None:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(outputs, y_val_tensor)
        return loss.item()

    def predict(self, X_test):
        self.network.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_test_tensor)
            probas = torch.sigmoid(outputs).cpu().numpy()
        return (probas > self.best_threshold).astype(int).reshape(-1)
        
    def predict_proba(self, X_test):
        self.network.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_test_tensor)
            probas = torch.sigmoid(outputs).cpu().numpy()
        return probas.reshape(-1)
        
    def analyze_feature_importance(self):
        """
        使用输入特征的梯度来估计特征重要性
        """
        if self.network is None:
            raise ValueError("模型未训练")
            
        self.network.eval()
        importance = np.zeros(len(self.feature_names))
        
        # 使用一小部分数据计算梯度
        sample_size = min(1000, len(self.feature_names))
        X_sample = np.random.randn(sample_size, len(self.feature_names))
        X_tensor = torch.FloatTensor(self.scaler.transform(X_sample)).to(self.device)
        X_tensor.requires_grad = True
        
        # 计算梯度
        output = self.network(X_tensor)
        output.sum().backward()
        
        # 使用梯度的绝对值平均作为特征重要性
        importance = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
        
        # 输出特征重要性
        for name, imp in zip(self.feature_names, importance):
            print(f"特征 {name}: {imp:.4f}")

    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.network is None:
            raise ValueError("模型未训练")
        
        logging.info("\n=== Model Evaluation ===")

        
        # 使用训练好的网络进行预测
        self.network.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_test_tensor)
            y_pred_proba = torch.sigmoid(outputs).cpu().numpy()
            y_pred = (y_pred_proba > self.best_threshold).astype(int).reshape(-1)
        
        # 计算评估指标
        test_metrics = self.metrics.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return test_metrics

    def evaluate_f1(self, X_val, y_val):
        """评估F1分数"""
        self.network.eval()
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_val_tensor)
            y_pred_proba = torch.sigmoid(outputs).cpu().numpy()
            y_pred = (y_pred_proba > self.best_threshold).astype(int).reshape(-1)
            return f1_score(y_val, y_pred)

    def get_parameters(self):
        """获取模型参数"""
        if self.network is None:
            raise ValueError("模型未初始化")
        return {
            name: param.data.clone().detach()
            for name, param in self.network.named_parameters()
        }

    def set_parameters(self, parameters):
        """设置模型参数"""
        if self.network is None:
            raise ValueError("模型未初始化")
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if name in parameters:
                    param.data = parameters[name].clone().detach()
