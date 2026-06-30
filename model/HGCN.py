import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)



class GCN_Layer(nn.Module):
    def __init__(self,num_of_features,num_of_filter):
        """One layer of GCN
        
        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCN_Layer,self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features = num_of_features,
                    out_features = num_of_filter),
            nn.ReLU()
        )
    def forward(self,input,adj):
        """CN
        
        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix, shape (N,N)
        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        batch_size,_,_ = input.shape  # (batch_size,N,D1) 
        adj = torch.from_numpy(adj).to(input.device)
        adj = adj.repeat(batch_size,1,1)
        input = torch.bmm(adj, input)  # (b,n,m) (b,m,p) ?(b,n,p)
        output = self.gcn_layer(input)
        return output

class STGeoModule(nn.Module):
    def __init__(self,grid_in_channel,num_of_gru_layers,seq_len,
                gru_hidden_size,num_of_target_time_feature):
        """[summary]
        
        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_gru_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature, 24(hour)+7(week)+1(holiday)=32
        """
        super(STGeoModule,self).__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=grid_in_channel,kernel_size=3,padding=1),
            nn.ReLU(),
        )

        self.grid_gru = nn.GRU(grid_in_channel,gru_hidden_size,num_of_gru_layers,batch_first=True)

        self.grid_att_fc1 = nn.Linear(in_features=gru_hidden_size,out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature,out_features=seq_len)
        self.grid_att_bias = nn.Parameter(torch.zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)

    def forward(self,grid_input,target_time_feature):
        """
        Arguments:
            grid_input {Tensor} -- grid input, shape: (batch_size,seq_len,D,W,H)
            target_time_feature {Tensor} -- the feature of target time, shape: (batch_size,num_target_time_feature)
        Returns:
            {Tensor} -- shape: (batch_size,hidden_size,W,H)
        """
        batch_size,T,D,W,H = grid_input.shape
        
        grid_input = grid_input.view(-1,D,W,H)  # reshape
        conv_output = self.grid_conv(grid_input)  # padding = 1: (batch_size*seq_len, D, W, H)
        
        conv_output = conv_output.view(batch_size,-1,D,W,H)\
                        .permute(0,3,4,1,2)\
                        .contiguous()\
                        .view(-1,T,D)
        gru_output,_ = self.grid_gru(conv_output)  # _: hn

        grid_target_time = torch.unsqueeze(target_time_feature,1).repeat(1,W*H,1).view(batch_size*W*H,-1)  # unsqueeze(x, dimension) repeat(m, n)
        grid_att_fc1_output = torch.squeeze(self.grid_att_fc1(gru_output))  # (W*H*Bs, T)
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)
        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output+grid_att_fc2_output+self.grid_att_bias))
        grid_att_score = grid_att_score.view(batch_size*W*H,-1,1)
        grid_output = torch.sum(gru_output * grid_att_score, dim=1)  # dot
        
        grid_output = grid_output.view(batch_size,W,H,-1).permute(0,3,1,2).contiguous()
    
        return grid_output


class STGeoLSTMModule(nn.Module):
    def __init__(self,grid_in_channel,num_of_lstm_layers,seq_len,
                lstm_hidden_size,num_of_target_time_feature):
        super(STGeoLSTMModule,self).__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=grid_in_channel,kernel_size=3,padding=1),
            nn.ReLU(),
        )

        self.grid_lstm = nn.LSTM(
            grid_in_channel,lstm_hidden_size,num_of_lstm_layers,batch_first=True
        )

        self.grid_att_fc1 = nn.Linear(in_features=lstm_hidden_size,out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature,out_features=seq_len)
        self.grid_att_bias = nn.Parameter(torch.zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)

    def forward(self,grid_input,target_time_feature):
        batch_size,T,D,W,H = grid_input.shape

        grid_input = grid_input.view(-1,D,W,H)
        conv_output = self.grid_conv(grid_input)

        conv_output = conv_output.view(batch_size,-1,D,W,H)\
                        .permute(0,3,4,1,2)\
                        .contiguous()\
                        .view(-1,T,D)
        lstm_output,_ = self.grid_lstm(conv_output)

        grid_target_time = torch.unsqueeze(target_time_feature,1).repeat(1,W*H,1).view(batch_size*W*H,-1)
        grid_att_fc1_output = torch.squeeze(self.grid_att_fc1(lstm_output))
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)
        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output+grid_att_fc2_output+self.grid_att_bias))
        grid_att_score = grid_att_score.view(batch_size*W*H,-1,1)
        grid_output = torch.sum(lstm_output * grid_att_score, dim=1)

        grid_output = grid_output.view(batch_size,W,H,-1).permute(0,3,1,2).contiguous()
        return grid_output


class STSemModule(nn.Module):
    def __init__(self,num_of_graph_feature,nums_of_graph_filters,
                seq_len,num_of_gru_layers,gru_hidden_size,
                num_of_target_time_feature,north_south_map,west_east_map):
        """
        Arguments:
            num_of_graph_feature {int} -- the number of graph node feature?batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            seq_len {int} -- the time length of input
            num_of_gru_layers {int} -- the number of GRU layers
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature24(hour)+7(week)+1(holiday)=32
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data

        """
        super(STSemModule,self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        # 2 GCN layer: 3-64-64
        self.road_gcn = nn.ModuleList()  # disorder list
        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.road_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))
        
        self.risk_gcn = nn.ModuleList()
        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.risk_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))

        self.poi_gcn = nn.ModuleList()
        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.poi_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))

        self.graph_gru = nn.GRU(num_of_filter,gru_hidden_size,num_of_gru_layers,batch_first=True)

        self.graph_att_fc1 = nn.Linear(in_features=gru_hidden_size,out_features=1)
        self.graph_att_fc2 = nn.Linear(in_features=num_of_target_time_feature,out_features=seq_len)
        self.graph_att_bias = nn.Parameter(torch.zeros(1))
        self.graph_att_softmax = nn.Softmax(dim=-1)


    def forward(self,graph_feature,road_adj,risk_adj,poi_adj,
                target_time_feature,grid_node_map):
        """
        Arguments:
            graph_feature {Tensor} -- Graph signal matrix?batch_size,T,D1,N)
            road_adj {np.array} -- road adjacent matrixhape?N,N)
            risk_adj {np.array} -- risk adjacent matrixhape?N,N)
            poi_adj {np.array} -- poi adjacent matrixhape?N,N)
            target_time_feature {Tensor} -- the feature of target timehape?batch_size,num_target_time_feature)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)
        Returns:
            {Tensor} -- shape?batch_size,pre_len,north_south_map,west_east_map)
        """        
        batch_size,T,D1,N = graph_feature.shape
        
        road_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output,road_adj)
        
        risk_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output,risk_adj)
        
        graph_output = road_graph_output + risk_graph_output

        if poi_adj is not None:
            poi_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
            for gcn_layer in self.poi_gcn:
                poi_graph_output = gcn_layer(poi_graph_output,poi_adj)
            graph_output += poi_graph_output

        graph_output = graph_output.view(batch_size,T,N,-1)\
                                    .permute(0,2,1,3)\
                                    .contiguous()\
                                    .view(batch_size*N,T,-1)  # (batch_size*N:32*243, T:7, 64)
        graph_output,_ = self.graph_gru(graph_output)  # (batch_size*N:32*243, T:7, 256)
        
        graph_target_time = torch.unsqueeze(target_time_feature,1).repeat(1,N,1).view(batch_size*N,-1)
        graph_att_fc1_output = torch.squeeze(self.graph_att_fc1(graph_output))
        graph_att_fc2_output = self.graph_att_fc2(graph_target_time)
        graph_att_score = self.graph_att_softmax(F.relu(graph_att_fc1_output+graph_att_fc2_output+self.graph_att_bias))
        graph_att_score = graph_att_score.view(batch_size*N,-1,1)
        graph_output = torch.sum(graph_output * graph_att_score,dim=1)
        graph_output = graph_output.view(batch_size,N,-1).contiguous()

        grid_node_map_tmp = torch.from_numpy(grid_node_map)\
                            .to(graph_feature.device)\
                            .repeat(batch_size,1,1)
        graph_output = torch.bmm(grid_node_map_tmp,graph_output)\
                            .permute(0,2,1)\
                            .view(batch_size,-1,self.north_south_map,self.west_east_map)
        return graph_output


def _normalize_adj_tensor(adj, device):
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj.astype(np.float32))
    adj = adj.float().to(device)
    eye = torch.eye(adj.shape[0], device=device)
    adj = torch.where(eye.bool(), torch.maximum(adj, eye), adj)
    return adj / torch.clamp(adj.sum(dim=-1, keepdim=True), min=1e-6)


def _grid_to_node_features(grid_features, grid_node_map):
    """Map grid features to valid graph nodes.

    grid_features: (C,W,H)
    grid_node_map: (W*H,N)
    returns: (N,C)
    """
    C, W, H = grid_features.shape
    grid_map = torch.from_numpy(grid_node_map).float().to(grid_features.device)
    flat = grid_features.view(C, W * H).permute(1, 0).contiguous()
    node_mass = torch.clamp(grid_map.sum(dim=0, keepdim=True).t(), min=1e-6)
    return torch.mm(grid_map.t(), flat) / node_mass


def _node_to_grid_features(node_features, grid_node_map, north_south_map, west_east_map):
    """Map node features back to grid.

    node_features: (B,N,C)
    returns: (B,C,W,H)
    """
    batch_size = node_features.shape[0]
    grid_map = torch.from_numpy(grid_node_map).float().to(node_features.device)
    grid_map = grid_map.unsqueeze(0).repeat(batch_size, 1, 1)
    grid_output = torch.bmm(grid_map, node_features)
    return grid_output.permute(0, 2, 1).contiguous().view(
        batch_size, -1, north_south_map, west_east_map
    )


def _grid_sequence_to_node_features(grid_features, grid_node_map):
    """Map a grid feature sequence to graph nodes.

    grid_features: (B,T,C,W,H)
    returns: (B,T,C,N)
    """
    batch_size, seq_len, channels, width, height = grid_features.shape
    grid_map = torch.from_numpy(grid_node_map).float().to(grid_features.device)
    flat = grid_features.view(batch_size * seq_len, channels, width * height)\
                        .permute(0, 2, 1)\
                        .contiguous()
    node_mass = torch.clamp(grid_map.sum(dim=0).view(1, -1, 1), min=1e-6)
    node_features = torch.matmul(grid_map.t().unsqueeze(0), flat) / node_mass
    return node_features.view(batch_size, seq_len, -1, channels)\
                        .permute(0, 1, 3, 2)\
                        .contiguous()


def _infer_nyc_like_indices(num_channels):
    if num_channels >= 48:
        return {
            "risk": [0],
            "poi": list(range(33, 40)),
            "weather": list(range(40, 46)),
            "flow": [46, 47],
        }
    return {
        "risk": [0],
        "poi": [],
        "weather": list(range(33, 39)),
        "flow": [39, 40],
    }


class RelationGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RelationGCN, self).__init__()
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            in_dim = input_dim if idx == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))

    def forward(self, node_input, adj):
        output = node_input
        for layer in self.layers:
            output = torch.bmm(adj, output)
            output = F.relu(layer(output))
        return output


class StructureBiasedGraphTransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, max_relations=4, dropout=0.1):
        super(StructureBiasedGraphTransformerLayer, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relation_bias = nn.Parameter(torch.zeros(num_heads, max_relations))
        self.last_attention = None
        self.last_relation_bias = None

    def forward(self, node_input, support_adjs):
        hidden = self.input_proj(node_input)
        batch_size, num_nodes, hidden_dim = hidden.shape
        q = self.q_proj(hidden).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_proj(hidden).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_proj(hidden).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        num_relations = len(support_adjs)
        relation_bias = self.relation_bias[:, :num_relations]
        stacked_adj = torch.stack(support_adjs, dim=1).unsqueeze(1)
        score = score + torch.sum(
            relation_bias.view(1, self.num_heads, num_relations, 1, 1) * stacked_adj,
            dim=2,
        )
        attention = F.softmax(score, dim=-1)
        context = torch.matmul(self.dropout(attention), v).permute(0, 2, 1, 3)\
            .contiguous()\
            .view(batch_size, num_nodes, hidden_dim)
        hidden = self.norm1(hidden + self.dropout(self.out_proj(context)))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        self.last_attention = attention.detach()
        self.last_relation_bias = relation_bias.detach()
        return hidden


class TrafficPropagationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len,
                num_of_gru_layers, num_of_target_time_feature,
                temporal_hidden_dim=None):
        super(TrafficPropagationModule, self).__init__()
        temporal_hidden_dim = temporal_hidden_dim or hidden_dim
        self.road_gcn = RelationGCN(input_dim, hidden_dim, num_layers)
        self.risk_gcn = RelationGCN(input_dim, hidden_dim, num_layers)
        self.functional_gcn = RelationGCN(input_dim, hidden_dim, num_layers)
        self.transformer_layers = nn.ModuleList()
        for idx in range(num_layers):
            layer_input_dim = input_dim if idx == 0 else hidden_dim
            self.transformer_layers.append(
                StructureBiasedGraphTransformerLayer(
                    layer_input_dim,
                    hidden_dim,
                    num_heads=4,
                    max_relations=4,
                    dropout=0.1,
                )
            )
        self.last_relation_weight = None
        self.last_attention = None
        self.propagation_backbone = "gcn"
        self.use_graph_gru = True
        self.graph_gru = nn.GRU(
            hidden_dim, temporal_hidden_dim, num_of_gru_layers, batch_first=True
        )
        self.no_gru_projection = nn.Linear(hidden_dim, temporal_hidden_dim)
        self.att_fc1 = nn.Linear(temporal_hidden_dim, 1)
        self.att_fc2 = nn.Linear(num_of_target_time_feature, seq_len)
        self.att_bias = nn.Parameter(torch.zeros(1))

    def _attend_sequence(self, graph_output, target_time_feature, att_fc1, att_fc2, att_bias):
        batch_nodes, seq_len, _ = graph_output.shape
        num_nodes = batch_nodes // target_time_feature.shape[0]
        graph_target_time = torch.unsqueeze(target_time_feature, 1)\
                                .repeat(1, num_nodes, 1)\
                                .view(batch_nodes, -1)
        att_fc1_output = torch.squeeze(att_fc1(graph_output))
        att_fc2_output = att_fc2(graph_target_time)
        att_score = F.softmax(
            F.relu(att_fc1_output + att_fc2_output + att_bias), dim=-1
        )
        att_score = att_score.view(batch_nodes, seq_len, 1)
        return torch.sum(graph_output * att_score, dim=1)

    def forward(
        self,
        traffic_feature,
        local_adj,
        risk_adj,
        functional_adj,
        target_time_feature,
        weather_adj=None,
    ):
        batch_size, seq_len, feature_dim, num_nodes = traffic_feature.shape
        traffic_input = traffic_feature.view(-1, feature_dim, num_nodes)\
                                        .permute(0, 2, 1)\
                                        .contiguous()
        local_adj = local_adj.unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
        risk_adj = risk_adj.unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
        functional_adj = functional_adj.unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
        if weather_adj is not None:
            if weather_adj.dim() == 2:
                weather_adj = weather_adj.unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
            else:
                weather_adj = weather_adj.unsqueeze(1)\
                                         .repeat(1, seq_len, 1, 1)\
                                         .view(batch_size * seq_len, num_nodes, num_nodes)
        if self.propagation_backbone == "gt":
            support_adjs = [local_adj, risk_adj, functional_adj]
            if weather_adj is not None:
                support_adjs.append(weather_adj)
            graph_output = traffic_input
            for transformer_layer in self.transformer_layers:
                graph_output = transformer_layer(graph_output, support_adjs)
            self.last_attention = self.transformer_layers[-1].last_attention
            self.last_relation_weight = self.transformer_layers[-1].last_relation_bias
        else:
            road_output = self.road_gcn(traffic_input, local_adj)
            risk_output = self.risk_gcn(traffic_input, risk_adj)
            functional_output = self.functional_gcn(traffic_input, functional_adj)
            relation_outputs = [road_output, risk_output, functional_output]
            if weather_adj is not None:
                weather_output = self.functional_gcn(traffic_input, weather_adj)
                relation_outputs.append(weather_output)
            graph_output = torch.stack(relation_outputs, dim=0).mean(dim=0)
            self.last_attention = None
            self.last_relation_weight = None
        graph_output = graph_output.view(batch_size, seq_len, num_nodes, -1)\
                                   .permute(0, 2, 1, 3)\
                                   .contiguous()\
                                   .view(batch_size * num_nodes, seq_len, -1)

        if self.use_graph_gru:
            graph_output, _ = self.graph_gru(graph_output)
        else:
            graph_output = self.no_gru_projection(graph_output)
        graph_output = self._attend_sequence(
            graph_output,
            target_time_feature,
            self.att_fc1,
            self.att_fc2,
            self.att_bias,
        )
        return graph_output.view(batch_size, num_nodes, -1).contiguous()


class WeatherLagModule(nn.Module):
    def __init__(self, weather_dim, hidden_dim, seq_len):
        super(WeatherLagModule, self).__init__()
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lag_score = nn.Linear(hidden_dim, 1)
        self.seq_bias = nn.Parameter(torch.zeros(seq_len))
        self.last_lag_weight = None

    def forward(self, weather_feature):
        batch_size, seq_len, weather_dim, num_nodes = weather_feature.shape
        weather_input = weather_feature.permute(0, 3, 1, 2)\
                                       .contiguous()\
                                       .view(batch_size * num_nodes, seq_len, weather_dim)
        weather_hidden = self.weather_encoder(weather_input)
        score = torch.squeeze(self.lag_score(weather_hidden), -1) + self.seq_bias
        lag_weight = F.softmax(score, dim=-1).unsqueeze(-1)
        self.last_lag_weight = lag_weight.detach()
        weather_message = torch.sum(weather_hidden * lag_weight, dim=1)
        return weather_message.view(batch_size, num_nodes, -1).contiguous()


class WeatherConditionalGraphModule(nn.Module):
    def __init__(self, weather_dim, hidden_dim, seq_len, graph_dim, top_k=12,
                history_len=24):
        super(WeatherConditionalGraphModule, self).__init__()
        self.top_k = top_k
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lag_score = nn.Linear(hidden_dim, 1)
        self.seq_bias = nn.Parameter(torch.zeros(history_len))
        self.activation_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + weather_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.susceptibility_head = nn.Sequential(
            nn.Linear(graph_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.support_logits = nn.Parameter(torch.tensor([0.0, 0.5, 0.0]))
        with torch.no_grad():
            self.activation_gate[-1].bias.fill_(-1.0)
            self.susceptibility_head[-1].bias.zero_()
        self.last_lag_weight = None
        self.last_weather_gate = None
        self.last_weather_adj = None
        self.last_support_weight = None

    def forward(
        self,
        weather_feature,
        local_adj,
        risk_adj,
        functional_adj,
        traffic_feature,
        static_prior,
        weather_history=None,
    ):
        weather_source = weather_history if weather_history is not None else weather_feature
        batch_size, history_len, weather_dim, num_nodes = weather_source.shape
        city_weather = weather_source.mean(dim=-1).contiguous()
        weather_hidden = self.weather_encoder(city_weather)
        if history_len <= self.seq_bias.shape[0]:
            seq_bias = self.seq_bias[:history_len]
        else:
            pad = self.seq_bias[-1:].repeat(history_len - self.seq_bias.shape[0])
            seq_bias = torch.cat([self.seq_bias, pad], dim=0)
        score = torch.squeeze(self.lag_score(weather_hidden), -1) + seq_bias
        lag_weight = F.softmax(score, dim=-1).unsqueeze(-1)
        weather_context = torch.sum(weather_hidden * lag_weight, dim=1)

        last_hidden = weather_hidden[:, -1, :]
        raw_mean = city_weather.mean(dim=1)
        raw_last = city_weather[:, -1, :]
        raw_delta = raw_last - raw_mean
        event_input = torch.cat(
            [weather_context, last_hidden, raw_last, raw_delta], dim=-1
        )
        event_gate = torch.sigmoid(self.activation_gate(event_input)).unsqueeze(1)

        traffic_recent = traffic_feature[:, -1, :, :].permute(0, 2, 1).contiguous()
        traffic_mean = traffic_feature.mean(dim=1).permute(0, 2, 1).contiguous()
        if static_prior.dim() == 2:
            prior_signal = static_prior[:, :1].unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            prior_signal = static_prior[:, :, :1]
        susceptibility_input = torch.cat(
            [traffic_recent, traffic_mean, prior_signal], dim=-1
        )
        node_susceptibility = torch.sigmoid(
            self.susceptibility_head(susceptibility_input)
        )
        node_gate = event_gate * node_susceptibility

        support_weight = F.softmax(self.support_logits, dim=0)
        support_adj = (
            support_weight[0] * local_adj
            + support_weight[1] * risk_adj
            + support_weight[2] * functional_adj
        ).clamp(min=0)
        eye = torch.eye(num_nodes, device=weather_feature.device, dtype=torch.bool)
        support_adj = support_adj.masked_fill(eye, 0.0)
        support_adj = support_adj / torch.clamp(
            support_adj.sum(dim=-1, keepdim=True), min=1e-6
        )

        pair_gate = 0.5 * (node_gate + node_gate.transpose(1, 2))
        edge_score = support_adj.unsqueeze(0) * pair_gate

        top_k = min(max(int(self.top_k), 0), max(num_nodes - 1, 0))
        if top_k == 0:
            weather_adj = torch.zeros_like(edge_score)
        else:
            topk_values, topk_indices = torch.topk(edge_score, top_k, dim=-1)
            weather_adj = torch.zeros_like(edge_score)
            weather_adj.scatter_(2, topk_indices, topk_values)

        self.last_lag_weight = lag_weight.detach()
        self.last_weather_gate = node_gate.detach()
        self.last_weather_adj = weather_adj.detach()
        self.last_support_weight = support_weight.detach()
        return weather_adj, node_gate


class AdaptiveFeatureRoleModule(nn.Module):
    def __init__(self, num_channels, propagation_dim, hidden_dim, pre_len,
                top_k=12, heuristic_scale=2.0):
        super(AdaptiveFeatureRoleModule, self).__init__()
        self.top_k = top_k
        self.heuristic_scale = heuristic_scale
        self.propagation_projector = nn.Linear(num_channels, propagation_dim)
        self.node_prior_head = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pre_len),
        )
        self.conditional_prior_head = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pre_len),
        )
        self.register_buffer("global_descriptor", torch.empty(0))
        self.last_role_weight = None
        self.last_conditional_gate = None
        self.last_functional_adj = None
        self.exclude_risk_in_functional_adj = False
        self.use_conditional_local_gate = True
        self.use_prior_outputs = True
        self.role_formula = "legacy"
        self.conditional_shift_tau = 0.25
        self.conditional_shift_scale = 8.0
        self.dvp_propagation_lambda = 1.0
        self.conditional_gate_mode = "legacy_suppress"
        self.conditional_gate_strength = 1.0
        self.conditional_gate_indices = []
        self.risk_channel_index = 0

    def set_global_descriptor(self, descriptor):
        self.global_descriptor = descriptor.detach().float()

    def _normalize_descriptor(self, descriptor):
        amplitude = torch.clamp(descriptor[:, :, 0:1].abs(), min=1e-4)
        amplitude_norm = descriptor[:, :, 0:1] / torch.clamp(
            descriptor[:, :, 0:1].mean(dim=1, keepdim=True), min=1e-6
        )
        return torch.cat(
            [
                amplitude_norm,
                descriptor[:, :, 1:2] / amplitude,
                descriptor[:, :, 2:3] / amplitude,
                descriptor[:, :, 3:4] / amplitude,
            ],
            dim=-1,
        )

    def _build_functional_adj(self, propagation_embedding, local_adj=None, risk_adj=None):
        num_nodes = propagation_embedding.shape[0]
        norm_embedding = F.normalize(propagation_embedding, p=2, dim=-1)
        functional_adj = torch.mm(norm_embedding, norm_embedding.t()).clamp(min=0)
        eye = torch.eye(num_nodes, device=propagation_embedding.device, dtype=torch.bool)
        functional_adj = functional_adj.masked_fill(eye, 0.0)
        if local_adj is not None:
            functional_adj = functional_adj.masked_fill(local_adj > 0, 0.0)
        if risk_adj is not None:
            functional_adj = functional_adj.masked_fill(risk_adj > 0, 0.0)
        top_k = min(max(int(self.top_k), 0), max(num_nodes - 1, 0))
        if top_k > 0 and top_k < num_nodes:
            topk_values, topk_indices = torch.topk(functional_adj, top_k, dim=-1)
            sparse_adj = torch.zeros_like(functional_adj)
            sparse_adj.scatter_(1, topk_indices, topk_values)
            functional_adj = torch.maximum(sparse_adj, sparse_adj.t())
        return functional_adj / torch.clamp(
            functional_adj.sum(dim=-1, keepdim=True), min=1e-6
        )

    def forward(self, grid_input, grid_node_map, local_adj=None, risk_adj=None):
        node_sequence = _grid_sequence_to_node_features(grid_input, grid_node_map)
        batch_size, seq_len, num_channels, num_nodes = node_sequence.shape

        mean_abs = node_sequence.abs().mean(dim=(1, 3))
        temporal_std = node_sequence.std(dim=1, unbiased=False).mean(dim=-1)
        spatial_std = node_sequence.std(dim=-1, unbiased=False).mean(dim=1)
        recent_delta = (
            node_sequence[:, -1, :, :] - node_sequence.mean(dim=1)
        ).abs().mean(dim=-1)
        local_descriptor = torch.stack(
            [mean_abs, temporal_std, spatial_std, recent_delta], dim=-1
        )
        descriptor = local_descriptor
        if self.global_descriptor.numel() > 0:
            descriptor = self.global_descriptor.to(node_sequence.device)\
                .unsqueeze(0)\
                .repeat(batch_size, 1, 1)
        if self.role_formula in ("dvp", "dvp_soft") and descriptor.shape[-1] >= 4:
            amplitude = torch.clamp(descriptor[:, :, 0], min=1e-4)
            difference_raw = descriptor[:, :, 1] / amplitude
            variation_raw = descriptor[:, :, 2] / amplitude
            propagation_raw = torch.clamp(descriptor[:, :, 3], min=0.0)
            difference_scale = torch.clamp(
                difference_raw.mean(dim=1, keepdim=True), min=1e-6
            )
            variation_scale = torch.clamp(
                variation_raw.mean(dim=1, keepdim=True), min=1e-6
            )
            propagation_scale = torch.clamp(
                propagation_raw.mean(dim=1, keepdim=True), min=1e-6
            )
            difference_hint = difference_raw / difference_scale
            variation_hint = variation_raw / variation_scale
            propagation_evidence = propagation_raw / propagation_scale

            local_amplitude = torch.clamp(local_descriptor[:, :, 0], min=1e-4)
            recent_std = (local_descriptor[:, :, 1] / local_amplitude) / variation_scale
            variation_shift = torch.abs(recent_std - variation_hint) / torch.clamp(
                variation_hint, min=1e-6
            )
            shift_weight = torch.sigmoid(
                self.conditional_shift_scale
                * (variation_shift - self.conditional_shift_tau)
            )
            conditional_variation = (
                (1.0 - shift_weight) * variation_hint
                + shift_weight * recent_std
            )

            if self.role_formula == "dvp_soft":
                propagation_hint = variation_hint * (
                    1.0 + self.dvp_propagation_lambda * propagation_evidence
                )
            else:
                propagation_hint = variation_hint * propagation_evidence
            node_prior_hint = difference_hint / (
                1.0 + variation_hint + propagation_evidence
            )
            conditional_hint = conditional_variation / (
                1.0 + 0.5 * propagation_evidence
            )
            conditional_gate = torch.ones_like(conditional_hint)
        else:
            descriptor_norm = self._normalize_descriptor(descriptor)
            local_descriptor_norm = self._normalize_descriptor(local_descriptor)

            temporal_hint = descriptor_norm[:, :, 1] + descriptor_norm[:, :, 3]
            spatial_hint = descriptor_norm[:, :, 2]
            spatiotemporal_hint = spatial_hint * temporal_hint
            propagation_hint = temporal_hint + 0.7 * spatiotemporal_hint
            node_prior_hint = spatial_hint / (1.0 + temporal_hint)
            conditional_hint = temporal_hint / (1.0 + spatial_hint)

            local_temporal_hint = local_descriptor_norm[:, :, 1] + local_descriptor_norm[:, :, 3]
            local_spatial_hint = local_descriptor_norm[:, :, 2]
            local_condition_hint = local_temporal_hint / (1.0 + local_spatial_hint)
            if self.use_conditional_local_gate:
                if self.conditional_gate_mode == "weather_anomaly_boost":
                    temporal_shift = torch.abs(local_temporal_hint - temporal_hint) / torch.clamp(
                        temporal_hint.abs(), min=1e-6
                    )
                    trigger = torch.sigmoid(
                        self.conditional_shift_scale
                        * (temporal_shift - self.conditional_shift_tau)
                    )
                    conditional_gate = torch.ones_like(local_condition_hint)
                    if self.conditional_gate_indices:
                        valid_indices = [
                            idx for idx in self.conditional_gate_indices
                            if 0 <= idx < conditional_gate.shape[1]
                        ]
                        if valid_indices:
                            conditional_gate[:, valid_indices] = (
                                1.0
                                + self.conditional_gate_strength
                                * trigger[:, valid_indices]
                            )
                else:
                    conditional_gate = local_condition_hint / (1.0 + local_condition_hint)
            else:
                conditional_gate = torch.ones_like(local_condition_hint)
            if self.role_formula == "two_role_legacy":
                node_prior_hint = torch.maximum(node_prior_hint, conditional_hint)
                conditional_hint = torch.full_like(conditional_hint, -1e6)
                conditional_gate = torch.ones_like(local_condition_hint)
        heuristic_logits = torch.stack(
            [propagation_hint, node_prior_hint, conditional_hint], dim=-1
        )
        role_weight = F.softmax(
            self.heuristic_scale * heuristic_logits, dim=-1
        )
        propagation_weight = role_weight[:, :, 0]
        node_prior_weight = role_weight[:, :, 1]
        conditional_prior_weight = role_weight[:, :, 2]

        node_sequence_last = node_sequence.permute(0, 1, 3, 2).contiguous()
        propagation_input = node_sequence_last * propagation_weight[:, None, None, :]
        propagation_feature = self.propagation_projector(propagation_input)\
            .permute(0, 1, 3, 2)\
            .contiguous()

        channel_profile = node_sequence.mean(dim=1).permute(0, 2, 1).contiguous()
        propagation_profile = channel_profile * propagation_weight[:, None, :]
        functional_profile = propagation_profile
        if (
            self.exclude_risk_in_functional_adj
            and 0 <= self.risk_channel_index < functional_profile.shape[-1]
        ):
            functional_profile = functional_profile.clone()
            functional_profile[:, :, self.risk_channel_index] = 0.0
        node_prior_profile = channel_profile * node_prior_weight[:, None, :]
        conditional_prior_profile = (
            channel_profile
            * conditional_prior_weight[:, None, :]
            * conditional_gate[:, None, :]
        )
        node_prior = self.node_prior_head(node_prior_profile)
        conditional_prior = self.conditional_prior_head(conditional_prior_profile)
        propagation_embedding = functional_profile.mean(dim=0)
        functional_adj = self._build_functional_adj(
            propagation_embedding, local_adj=local_adj, risk_adj=risk_adj
        )

        self.last_role_weight = role_weight.detach()
        self.last_conditional_gate = conditional_gate.detach()
        self.last_functional_adj = functional_adj.detach()
        return propagation_feature, functional_adj, node_prior, conditional_prior, role_weight


class StaticPOIModule(nn.Module):
    def __init__(self, poi_dim, hidden_dim, pre_len, num_clusters=8,
                kmeans_steps=12, top_k=12):
        super(StaticPOIModule, self).__init__()
        self.poi_dim = poi_dim
        self.num_clusters = num_clusters
        self.kmeans_steps = kmeans_steps
        self.top_k = top_k
        self.cluster_embedding = nn.Embedding(num_clusters, hidden_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(poi_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pre_len),
        )
        self.register_buffer("cached_functional_adj", torch.empty(0))
        self.register_buffer("cached_cluster_ids", torch.empty(0).long())

    def _cluster_poi(self, poi_features):
        num_nodes = poi_features.shape[0]
        num_clusters = min(self.num_clusters, num_nodes)
        norm_poi = poi_features / torch.clamp(
            poi_features.sum(dim=-1, keepdim=True), min=1e-6
        )
        centers = norm_poi[:num_clusters].clone()
        for _ in range(self.kmeans_steps):
            diff = norm_poi.unsqueeze(1) - centers.unsqueeze(0)
            distance = torch.sum(diff * diff, dim=-1)
            cluster_ids = torch.argmin(distance, dim=1)
            new_centers = []
            for cluster_idx in range(num_clusters):
                mask = cluster_ids == cluster_idx
                if torch.sum(mask) == 0:
                    new_centers.append(centers[cluster_idx])
                else:
                    new_centers.append(norm_poi[mask].mean(dim=0))
            centers = torch.stack(new_centers, dim=0)
        diff = norm_poi.unsqueeze(1) - centers.unsqueeze(0)
        distance = torch.sum(diff * diff, dim=-1)
        cluster_ids = torch.argmin(distance, dim=1)
        if num_clusters < self.num_clusters:
            cluster_ids = cluster_ids.clamp(max=num_clusters - 1)
        return cluster_ids

    def build_static_graph(self, poi_features, local_adj=None, risk_adj=None):
        cluster_ids = self._cluster_poi(poi_features)
        norm_poi = poi_features / torch.clamp(
            poi_features.norm(p=2, dim=-1, keepdim=True), min=1e-6
        )
        functional_adj = torch.mm(norm_poi, norm_poi.t()).clamp(min=0)
        functional_adj.fill_diagonal_(0)
        if local_adj is not None:
            functional_adj = functional_adj.masked_fill(local_adj > 0, 0)
        if risk_adj is not None:
            functional_adj = functional_adj.masked_fill(risk_adj > 0, 0)
        if self.top_k is not None and self.top_k > 0 and self.top_k < functional_adj.shape[-1]:
            topk_values, topk_indices = torch.topk(functional_adj, self.top_k, dim=-1)
            sparse_adj = torch.zeros_like(functional_adj)
            sparse_adj.scatter_(1, topk_indices, topk_values)
            functional_adj = torch.maximum(sparse_adj, sparse_adj.t())
        functional_adj = functional_adj / torch.clamp(
            functional_adj.sum(dim=-1, keepdim=True), min=1e-6
        )
        return functional_adj, cluster_ids

    def forward(self, poi_features, local_adj=None, risk_adj=None):
        if self.cached_functional_adj.numel() == 0:
            with torch.no_grad():
                functional_adj, cluster_ids = self.build_static_graph(
                    poi_features.detach(),
                    local_adj.detach() if local_adj is not None else None,
                    risk_adj.detach() if risk_adj is not None else None,
                )
            self.cached_functional_adj = functional_adj
            self.cached_cluster_ids = cluster_ids.long()
        functional_adj = self.cached_functional_adj.to(poi_features.device)
        cluster_ids = self.cached_cluster_ids.to(poi_features.device)
        cluster_embed = self.cluster_embedding(cluster_ids)
        prior_input = torch.cat([poi_features, cluster_embed], dim=-1)
        static_prior = self.prior_head(prior_input)
        return functional_adj, static_prior


class DSHGNN(nn.Module):
    def __init__(self,grid_in_channel,num_of_gru_layers,seq_len,pre_len,
                gru_hidden_size,num_of_target_time_feature,
                num_of_graph_feature,nums_of_graph_filters,
                north_south_map,west_east_map):
        """Dynamic-static decoupled heterogeneous graph model.

        Traffic nodes are the prediction subjects. Road adjacency captures local
        diffusion, POI functional clusters provide non-local traffic edges and a
        static risk prior, and weather contributes optional lag-aware exogenous
        messages.
        """
        super(DSHGNN,self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.pre_len = pre_len
        self.grid_in_channel = grid_in_channel
        self.channel_indices = _infer_nyc_like_indices(grid_in_channel)
        gcn_hidden_dim = nums_of_graph_filters[-1]
        graph_temporal_dim = gru_hidden_size
        self.st_geo_module = STGeoLSTMModule(
            grid_in_channel,
            num_of_gru_layers,
            seq_len,
            gru_hidden_size,
            num_of_target_time_feature,
        )
        self.traffic_module = TrafficPropagationModule(
            num_of_graph_feature,
            gcn_hidden_dim,
            len(nums_of_graph_filters),
            seq_len,
            num_of_gru_layers,
            num_of_target_time_feature,
            temporal_hidden_dim=graph_temporal_dim,
        )
        self.weather_module = WeatherLagModule(
            len(self.channel_indices["weather"]), gcn_hidden_dim, seq_len
        )
        self.use_weather = True
        self.use_weather_subgraph = False
        self.use_weather_as_relation = False
        self.use_weather_rank_head = True
        self.use_adaptive_feature_roles = False
        self.use_poi_in_grid = False
        self.use_grid_branch = True
        self.use_graph_gru = True
        self.propagation_backbone = "gcn"
        self.exclude_risk_in_functional_adj = False
        self.use_conditional_local_gate = True
        self.use_prior_outputs = True
        self.role_formula = "legacy"
        self.conditional_shift_tau = 0.25
        self.conditional_shift_scale = 8.0
        self.dvp_propagation_lambda = 1.0
        self.conditional_gate_mode = "legacy_suppress"
        self.conditional_gate_strength = 1.0
        poi_dim = max(len(self.channel_indices["poi"]), 1)
        self.poi_module = StaticPOIModule(poi_dim, gcn_hidden_dim, pre_len)
        self.feature_role_module = AdaptiveFeatureRoleModule(
            grid_in_channel,
            num_of_graph_feature,
            gcn_hidden_dim,
            pre_len,
            top_k=12,
        )
        self.weather_to_traffic = nn.Linear(gcn_hidden_dim, graph_temporal_dim)
        self.weather_scale_logit = nn.Parameter(torch.tensor(-4.0))
        fusion_channel = 16
        self.grid_weight = nn.Conv2d(
            in_channels=gru_hidden_size,out_channels=fusion_channel,kernel_size=1
        )
        self.graph_weight = nn.Conv2d(
            in_channels=graph_temporal_dim,out_channels=fusion_channel,kernel_size=1
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(
                in_channels=gru_hidden_size + graph_temporal_dim,
                out_channels=fusion_channel,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )
        self.output_layer = nn.Linear(
            fusion_channel * north_south_map * west_east_map,
            pre_len * north_south_map * west_east_map
        )
        self.prior_scale = nn.Parameter(torch.tensor(0.01))
        self.conditional_prior_scale = nn.Parameter(torch.tensor(0.01))
        self.semantic_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.weather_rank_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, pre_len),
        )
        self.weather_rank_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.weather_graph_module = WeatherConditionalGraphModule(
            len(self.channel_indices["weather"]),
            gcn_hidden_dim,
            seq_len,
            num_of_graph_feature,
            top_k=12,
            history_len=24,
        )

    def forward(self,grid_input,target_time_feature,graph_feature,
                road_adj,risk_adj,poi_adj,grid_node_map,weather_history=None):
        """
        Arguments:
            grid_input {Tensor} -- grid input, shape: (batch_size,T,D,W,H)
            graph_feature {Tensor} -- Graph signal matrix, (batch_size,T,D1,N)
            target_time_feature {Tensor} -- the feature of target time, shape: (batch_size,num_target_time_feature)
            road_adj {np.array} -- road adjacent matrix, shape: (N,N)
            risk_adj {np.array} -- risk adjacent matrix, shape: (N,N)
            poi_adj {np.array} -- poi adjacent matrix, shape: (N,N)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)

        Returns:
            {Tensor} -- shape: (batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, seq_len, _, _, _ = grid_input.shape
        local_adj = _normalize_adj_tensor(road_adj, graph_feature.device)
        risk_pattern_adj = _normalize_adj_tensor(risk_adj, graph_feature.device)
        local_mask_adj = torch.from_numpy(road_adj.astype(np.float32)).to(graph_feature.device)
        risk_mask_adj = torch.from_numpy(risk_adj.astype(np.float32)).to(graph_feature.device)

        poi_indices = self.channel_indices["poi"]
        if len(poi_indices) > 0:
            poi_grid = grid_input[0, 0, poi_indices, :, :]
            poi_features = _grid_to_node_features(poi_grid, grid_node_map)
        else:
            poi_features = torch.zeros(
                graph_feature.shape[-1], 1, device=graph_feature.device
            )
        functional_adj, static_prior = self.poi_module(
            poi_features, local_mask_adj, risk_mask_adj
        )
        if poi_adj is not None:
            semantic_adj = _normalize_adj_tensor(poi_adj, graph_feature.device)
            semantic_mix = torch.sigmoid(self.semantic_mix_logit)
            functional_adj = (
                semantic_mix * semantic_adj
                + (1.0 - semantic_mix) * functional_adj
            )
        traffic_feature = graph_feature
        conditional_prior = torch.zeros_like(static_prior)
        if self.use_adaptive_feature_roles:
            self.feature_role_module.exclude_risk_in_functional_adj = (
                self.exclude_risk_in_functional_adj
            )
            self.feature_role_module.use_conditional_local_gate = (
                self.use_conditional_local_gate
            )
            self.feature_role_module.role_formula = self.role_formula
            self.feature_role_module.conditional_shift_tau = self.conditional_shift_tau
            self.feature_role_module.conditional_shift_scale = (
                self.conditional_shift_scale
            )
            self.feature_role_module.dvp_propagation_lambda = (
                self.dvp_propagation_lambda
            )
            self.feature_role_module.conditional_gate_mode = (
                self.conditional_gate_mode
            )
            self.feature_role_module.conditional_gate_strength = (
                self.conditional_gate_strength
            )
            self.feature_role_module.conditional_gate_indices = (
                self.channel_indices["weather"]
            )
            (
                traffic_feature,
                functional_adj,
                static_prior,
                conditional_prior,
                _,
            ) = self.feature_role_module(
                grid_input,
                grid_node_map,
                local_adj=local_mask_adj,
                risk_adj=risk_mask_adj,
            )
        if not self.use_prior_outputs:
            static_prior = torch.zeros_like(static_prior)
            conditional_prior = torch.zeros_like(conditional_prior)

        weather_indices = self.channel_indices["weather"]
        weather_feature = None
        weather_adj = None
        weather_node_gate = None
        if len(weather_indices) > 0 and (self.use_weather or self.use_weather_subgraph):
            weather_feature = _grid_sequence_to_node_features(
                grid_input[:, :, weather_indices, :, :], grid_node_map
            )
        if self.use_weather_subgraph and weather_feature is not None:
            weather_history_feature = None
            if weather_history is not None:
                weather_history_feature = _grid_sequence_to_node_features(
                    weather_history, grid_node_map
                )
            weather_adj, weather_node_gate = self.weather_graph_module(
                weather_feature,
                local_adj,
                risk_pattern_adj,
                functional_adj,
                traffic_feature,
                static_prior,
                weather_history=weather_history_feature,
            )
            if not self.use_weather_as_relation:
                weather_adj = None

        self.traffic_module.use_graph_gru = self.use_graph_gru
        self.traffic_module.propagation_backbone = self.propagation_backbone
        traffic_output = self.traffic_module(
            traffic_feature,
            local_adj,
            risk_pattern_adj,
            functional_adj,
            target_time_feature,
            weather_adj=weather_adj,
        )

        node_hidden = traffic_output
        if self.use_weather and len(weather_indices) > 0:
            if weather_feature is None:
                weather_feature = _grid_sequence_to_node_features(
                    grid_input[:, :, weather_indices, :, :], grid_node_map
                )
            weather_message = self.weather_module(weather_feature)
            weather_scale = torch.sigmoid(self.weather_scale_logit)
            node_hidden = node_hidden + weather_scale * self.weather_to_traffic(weather_message)

        graph_grid = _node_to_grid_features(
            node_hidden, grid_node_map, self.north_south_map, self.west_east_map
        )
        if static_prior.dim() == 2:
            static_prior_batch = static_prior.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            static_prior_batch = static_prior
        static_prior_grid = _node_to_grid_features(
            static_prior_batch,
            grid_node_map,
            self.north_south_map,
            self.west_east_map,
        )
        if conditional_prior.dim() == 2:
            conditional_prior_batch = conditional_prior.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            conditional_prior_batch = conditional_prior
        conditional_prior_grid = _node_to_grid_features(
            conditional_prior_batch,
            grid_node_map,
            self.north_south_map,
            self.west_east_map,
        )
        graph_fusion = self.graph_weight(graph_grid)
        if self.use_grid_branch:
            grid_branch_input = grid_input
            if not self.use_poi_in_grid and len(poi_indices) > 0:
                grid_branch_input = grid_input.clone()
                grid_branch_input[:, :, poi_indices, :, :] = 0.0
            grid_output = self.st_geo_module(grid_branch_input,target_time_feature)
            grid_fusion = self.grid_weight(grid_output)
            fusion_gate = self.fusion_gate(torch.cat([grid_output, graph_grid], dim=1))
            fusion_output = fusion_gate * grid_fusion + (1.0 - fusion_gate) * graph_fusion
        else:
            fusion_output = graph_fusion
        final_output = self.output_layer(fusion_output.view(batch_size, -1))\
                           .view(batch_size, self.pre_len, self.north_south_map, self.west_east_map)
        final_output = final_output + self.prior_scale * static_prior_grid
        final_output = final_output + self.conditional_prior_scale * conditional_prior_grid
        if self.use_weather_rank_head and weather_node_gate is not None:
            weather_rank_prior = self.weather_rank_head(weather_node_gate)
            weather_rank_grid = _node_to_grid_features(
                weather_rank_prior,
                grid_node_map,
                self.north_south_map,
                self.west_east_map,
            )
            weather_rank_scale = torch.sigmoid(self.weather_rank_scale_logit)
            final_output = final_output + weather_rank_scale * weather_rank_grid
        return final_output


class OriginalGSNet(nn.Module):
    def __init__(self,grid_in_channel,num_of_gru_layers,seq_len,pre_len,
                gru_hidden_size,num_of_target_time_feature,
                num_of_graph_feature,nums_of_graph_filters,
                north_south_map,west_east_map):
        super(OriginalGSNet,self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map

        self.st_geo_module = STGeoModule(grid_in_channel,num_of_gru_layers,seq_len,
                                        gru_hidden_size,num_of_target_time_feature)

        self.st_sem_module = STSemModule(num_of_graph_feature,nums_of_graph_filters,
                                        seq_len,num_of_gru_layers,gru_hidden_size,
                                        num_of_target_time_feature,north_south_map,west_east_map)

        fusion_channel = 16
        self.grid_weigth = nn.Conv2d(in_channels=gru_hidden_size,out_channels=fusion_channel,kernel_size=1)
        self.graph_weigth = nn.Conv2d(in_channels=gru_hidden_size,out_channels=fusion_channel,kernel_size=1)
        self.output_layer = nn.Linear(fusion_channel*north_south_map*west_east_map,pre_len*north_south_map*west_east_map)

    def forward(self,grid_input,target_time_feature,graph_feature,
                road_adj,risk_adj,poi_adj,grid_node_map,weather_history=None):
        batch_size,_,_,_,_ = grid_input.shape

        grid_output = self.st_geo_module(grid_input,target_time_feature)
        graph_output = self.st_sem_module(graph_feature,road_adj,risk_adj,poi_adj,
                                        target_time_feature,grid_node_map)

        grid_output = self.grid_weigth(grid_output)
        graph_output = self.graph_weigth(graph_output)
        fusion_output = (grid_output + graph_output).view(batch_size,-1)
        final_output = self.output_layer(fusion_output)\
                            .view(batch_size,-1,self.north_south_map,self.west_east_map)
        return final_output


class GridOnlyNet(nn.Module):
    def __init__(self,grid_in_channel,num_of_gru_layers,seq_len,pre_len,
                gru_hidden_size,num_of_target_time_feature,
                num_of_graph_feature,nums_of_graph_filters,
                north_south_map,west_east_map):
        super(GridOnlyNet,self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.pre_len = pre_len
        self.channel_indices = _infer_nyc_like_indices(grid_in_channel)
        self.use_poi_in_grid = False
        self.st_geo_module = STGeoLSTMModule(
            grid_in_channel,
            num_of_gru_layers,
            seq_len,
            gru_hidden_size,
            num_of_target_time_feature,
        )
        self.output_layer = nn.Linear(
            gru_hidden_size * north_south_map * west_east_map,
            pre_len * north_south_map * west_east_map
        )

    def forward(self,grid_input,target_time_feature,graph_feature,
                road_adj,risk_adj,poi_adj,grid_node_map,weather_history=None):
        batch_size,_,_,_,_ = grid_input.shape
        poi_indices = self.channel_indices["poi"]
        grid_branch_input = grid_input
        if not self.use_poi_in_grid and len(poi_indices) > 0:
            grid_branch_input = grid_input.clone()
            grid_branch_input[:, :, poi_indices, :, :] = 0.0
        grid_output = self.st_geo_module(grid_branch_input,target_time_feature)
        final_output = self.output_layer(grid_output.view(batch_size, -1))\
                           .view(batch_size, self.pre_len, self.north_south_map, self.west_east_map)
        return final_output


class GSNet(DSHGNN):
    """Compatibility alias used by the existing training script."""
    pass
