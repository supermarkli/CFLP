import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os

class ExperimentVisualizer:
    """实验结果可视化工具类"""
    
    def __init__(self, save_dir='results/plots'):
        """
        初始化可视化工具
        
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_metrics_comparison(self, metrics_dict, title='模型性能对比'):
        """
        绘制不同模型的性能指标对比图
        
        Args:
            metrics_dict: 包含各模型指标的字典
            title: 图表标题
        """
        # 转换数据格式
        models = []
        metric_names = []
        values = []
        
        for model_name, metrics in metrics_dict.items():
            for metric_name, value in metrics.items():
                models.append(model_name)
                metric_names.append(metric_name)
                values.append(value)
                
        df = pd.DataFrame({
            'Model': models,
            'Metric': metric_names,
            'Value': values
        })
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Value', hue='Metric')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'{self.save_dir}/metrics_comparison.png')
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'{self.save_dir}/{model_name}_confusion_matrix.png')
        plt.close()
        
    def plot_training_history(self, history, model_name):
        """
        绘制训练历史
        
        Args:
            history: 训练历史数据
            model_name: 模型名称
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制训练集和验证集的损失
        if 'loss' in history and 'val_loss' in history:
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='训练集损失')
            plt.plot(history['val_loss'], label='验证集损失')
            plt.title('模型损失')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()
        
        # 绘制准确率
        if 'accuracy' in history and 'val_accuracy' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['accuracy'], label='训练集准确率')
            plt.plot(history['val_accuracy'], label='验证集准确率')
            plt.title('模型准确率')
            plt.xlabel('轮次')
            plt.ylabel('准确率')
            plt.legend()
            
        plt.suptitle(f'{model_name} 训练历史')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'{self.save_dir}/{model_name}_training_history.png')
        plt.close()
        
    def plot_federated_metrics(self, metrics_history, model_name):
        """
        绘制联邦学习训练过程中的指标变化
        
        Args:
            metrics_history: 包含每轮训练指标的列表
            model_name: 模型名称
        """
        # 转换数据格式
        rounds = list(range(1, len(metrics_history) + 1))
        metrics_df = pd.DataFrame(metrics_history)
        
        plt.figure(figsize=(12, 6))
        
        for metric in metrics_df.columns:
            plt.plot(rounds, metrics_df[metric], label=metric)
            
        plt.title(f'{model_name} 联邦学习训练过程')
        plt.xlabel('训练轮次')
        plt.ylabel('指标值')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'{self.save_dir}/{model_name}_federated_training.png')
        plt.close() 