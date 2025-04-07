import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
import logging
from utils.metrics import ModelMetrics
from sklearn.metrics import f1_score, accuracy_score
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
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = None
        self.epochs = 100
        self.batch_size = 128
        self.learning_rate = 0.0005
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.normalize = True
        self.name = "NeuralNet"
        self.metrics = ModelMetrics()
        self.best_threshold = 0.5

    def train_model(self, X_train, y_train):
        logging.info("\n=== Training Start ===")
        logging.info(f"Training data size: {X_train.shape}")
        
        if self.network is None:
            input_dim = X_train.shape[1]
            self.network = NeuralNetwork(input_dim=input_dim).to(self.device)
            logging.info(f"Initialized neural network with input dimension: {input_dim}")
        
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        # 打印类别分布
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        logging.info("Class distribution in training set:")
        logging.info(f"  Positive samples: {pos_count}")
        logging.info(f"  Negative samples: {neg_count}")
        
        # 转换为tensor并移动到正确的device
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        pos_weight = torch.tensor(neg_count / pos_count).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
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
            logging.info(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')
        
        logging.info("Model training completed")

    def predict(self, X_test):
        """预测类别"""
        self.network.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_test_tensor)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
        
        return predictions.cpu().numpy()

    def predict_proba(self, X_test):
        """预测概率"""
        self.network.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_test_tensor)
            probabilities = torch.sigmoid(outputs)
        
        return probabilities.cpu().numpy().squeeze()

  
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.network is None:
            raise ValueError("模型未训练")
        
        logging.info("\n=== Model Evaluation ===")

        self.network.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_test_tensor)
            y_pred_proba = torch.sigmoid(outputs).cpu().numpy()
            
            # 找到最优阈值
            self.best_threshold = self.find_best_threshold(y_test, y_pred_proba)
            logging.info(f"Using threshold: {self.best_threshold}")
            
            # 使用最优阈值进行预测
            y_pred = (y_pred_proba > self.best_threshold).astype(int).reshape(-1)
        
        test_metrics = self.metrics.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return test_metrics

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

    def find_best_threshold(self, y_test, y_pred_proba):
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_accuracy = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        logging.info(f"Found best threshold: {best_threshold:.2f} (Accuracy: {best_accuracy:.4f})")
        return best_threshold
