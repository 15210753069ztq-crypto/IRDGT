import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.metrics import mask_evaluation_np


class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min
        
        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,3,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def _scale_columns(self, data, columns):
        denom = self.max[columns] - self.min[columns]
        denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
        data[:,columns] = (data[:,columns] - self.min[columns]) / denom
    def transform(self, data):
        """norm train, valid, test
        
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        
        T,D,W,H = data.shape
        data = np.transpose(data,(0,2,3,1)).reshape((-1,D))  # (T*W*H, D)
        self._scale_columns(data, [0])
        self._scale_columns(data, list(range(33, 40)))
        self._scale_columns(data, [40, 46, 47])
        return np.transpose(data.reshape((T,W,H,-1)),(0,3,1,2))
    
    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min
        
        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,3,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def _scale_columns(self, data, columns):
        denom = self.max[columns] - self.min[columns]
        denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
        data[:,columns] = (data[:,columns] - self.min[columns]) / denom
    def transform(self, data):
        """norm trainalidest
        
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W,H = data.shape
        data = np.transpose(data,(0,2,3,1)).reshape((-1,D))#(T*W*H,D)
        self._scale_columns(data, [0, 33, 39, 40])
        return np.transpose(data.reshape((T,W,H,-1)),(0,3,1,2))
    
    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


def ranking_loss(predicts, labels, region_mask, margin=0.01, max_negatives=64):
    """Encourage accident grids to rank above non-accident grids."""
    batch_size, pre_len, _, _ = predicts.shape
    if isinstance(region_mask, np.ndarray):
        valid_mask = torch.from_numpy(region_mask).to(predicts.device).bool().view(-1)
    else:
        valid_mask = region_mask.to(predicts.device).bool().view(-1)
    pred_flat = predicts.reshape(batch_size * pre_len, -1)[:, valid_mask]
    label_flat = labels.reshape(batch_size * pre_len, -1)[:, valid_mask]

    losses = []
    for sample_pred, sample_label in zip(pred_flat, label_flat):
        pos_pred = sample_pred[sample_label > 0]
        neg_pred = sample_pred[sample_label <= 0]
        if pos_pred.numel() == 0 or neg_pred.numel() == 0:
            continue
        if max_negatives is not None and max_negatives > 0 and neg_pred.numel() > max_negatives:
            _, top_indices = torch.topk(neg_pred, max_negatives)
            neg_pred = neg_pred[top_indices]
        losses.append(F.relu(margin - pos_pred.unsqueeze(1) + neg_pred.unsqueeze(0)).mean())

    if not losses:
        return predicts.new_tensor(0.0)
    return torch.stack(losses).mean()


def mask_loss(predicts,labels,region_mask,data_type="nyc",
              ranking_loss_weight=0.0,
              ranking_margin=0.01,
              ranking_max_negatives=64,
              target_time_feature=None,
              high_frequency_loss_weight=0.0,
              high_frequency_hours=None):
    """
    
    Arguments:
        predicts {Tensor} -- predict, (batch_size,pre_len,W,H)
        labels {Tensor} -- label, (batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix, (W,H)
        data_type {str} -- nyc/chicago
    
    Returns:
        {Tensor} -- MSELoss,(1,)
    """
    batch_size,pre_len,_,_ = predicts.shape
    if isinstance(region_mask, np.ndarray):
        region_mask = torch.from_numpy(region_mask).to(predicts.device)
    else:
        region_mask = region_mask.to(predicts.device)
    region_mask /= region_mask.mean()  # ? region_mask.sum(): 50-400 (/ 0.125)
    loss = ((labels-predicts)*region_mask)**2
    if data_type=='nyc':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <=0
        index_2 = (labels > 0) & (labels <= 0.04)
        index_3 = (labels > 0.04) & (labels <= 0.08)
        index_4 = labels > 0.08
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    elif data_type=='chicago':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <=0
        index_2 = (labels > 0) & (labels <= 1/17)
        index_3 = (labels > 1/17) & (labels <= 2/17)
        index_4 = labels > 2/17
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask

    if (
        high_frequency_loss_weight > 0
        and target_time_feature is not None
        and target_time_feature.shape[-1] >= 24
    ):
        if high_frequency_hours is None:
            high_frequency_hours = [6, 7, 8, 15, 16, 17, 18]
        high_hours = torch.tensor(
            high_frequency_hours,
            device=predicts.device,
            dtype=torch.long,
        )
        hour_index = torch.argmax(target_time_feature[:, :24], dim=-1)
        high_mask = (hour_index.unsqueeze(-1) == high_hours.unsqueeze(0))\
            .any(dim=-1)\
            .float()
        sample_weight = 1.0 + high_frequency_loss_weight * high_mask
        sample_weight = sample_weight / torch.clamp(sample_weight.mean(), min=1e-6)
        loss = loss * sample_weight.view(batch_size, 1, 1, 1)

    base_loss = torch.mean(loss)
    if ranking_loss_weight <= 0:
        return base_loss
    return base_loss + ranking_loss_weight * ranking_loss(
        predicts,
        labels,
        region_mask,
        margin=ranking_margin,
        max_negatives=ranking_max_negatives,
    )

@torch.no_grad()
def compute_loss(net,dataloader,risk_mask,road_adj,risk_adj,poi_adj,
                grid_node_map,global_step,device,
                data_type='nyc',
                ranking_loss_weight=0.0,
                ranking_margin=0.01,
                ranking_max_negatives=64,
                high_frequency_loss_weight=0.0,
                high_frequency_hours=None):
    """Compute the same objective used for training."""
    net.eval()
    temp = []
    for batch in dataloader:
        if len(batch) == 5:
            feature,target_time,graph_feature,weather_history,label = batch
            weather_history = weather_history.to(device)
        else:
            feature,target_time,graph_feature,label = batch
            weather_history = None
        feature,target_time,graph_feature,label = feature.to(device),target_time.to(device),graph_feature.to(device),label.to(device)
        prediction = net(
            feature,
            target_time,
            graph_feature,
            road_adj,
            risk_adj,
            poi_adj,
            grid_node_map,
            weather_history=weather_history,
        )
        l = mask_loss(
            prediction,
            label,
            risk_mask,
            data_type,
            ranking_loss_weight=ranking_loss_weight,
            ranking_margin=ranking_margin,
            ranking_max_negatives=ranking_max_negatives,
            target_time_feature=target_time,
            high_frequency_loss_weight=high_frequency_loss_weight,
            high_frequency_hours=high_frequency_hours,
        )
        temp.append(l.cpu().item())
    return sum(temp) / len(temp)


@torch.no_grad()
def predict_and_evaluate(net,dataloader,risk_mask,road_adj,risk_adj,poi_adj,
                        grid_node_map,global_step,scaler,device):
    """predict val/test, return metrics
    
    Arguments:
        net {Model} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix, shape(W,H)
        road_adj  {np.array} -- road adjacent matrix, shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix, shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix, shape(N,N)
        global_step {int} -- global_step
        scaler {Scaler} -- record max and min
        device {Device} -- GPU
    
    Returns:
        np.float32 -- RMSE, Recall, MAP
        np.array -- label and pre, shape(num_sample,pre_len,W,H)

    """
    net.eval()
    prediction_list = []
    label_list = []
    for batch in dataloader:
        if len(batch) == 5:
            feature,target_time,graph_feature,weather_history,label = batch
            weather_history = weather_history.to(device)
        else:
            feature,target_time,graph_feature,label = batch
            weather_history = None
        feature,target_time,graph_feature,label = feature.to(device),target_time.to(device),graph_feature.to(device),label.to(device)
        prediction = net(
            feature,
            target_time,
            graph_feature,
            road_adj,
            risk_adj,
            poi_adj,
            grid_node_map,
            weather_history=weather_history,
        )
        prediction_list.append(prediction.cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_,recall_,map_ = mask_evaluation_np(inverse_trans_label,inverse_trans_pre,risk_mask,0)
    return rmse_,recall_,map_,inverse_trans_pre,inverse_trans_label

@torch.no_grad()
def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    if torch.is_tensor(h):  #?        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:  #
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()
