import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class eagle(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=1500, sfreq=500, 
                 dropoutRate=0.2, F1=16, D=2, gamma_F1=32, 
                 reduced_dim=48,
                 band_defs=((8,30), (30,60), (60,80))):
        super().__init__()
        self.sfreq = sfreq
        self.band_defs = band_defs
        
        # 다운샘플링을 4배로 줄임 (16배 → 4배)
        self.target_time_dim = Samples // 4

        # 주파수 대역별 커널 길이 (대략 주파수 하한의 1~2주기)
        def freq_kernel(low_f):
            cycles = 1.5
            k = int((cycles * sfreq) / low_f)
            return max(8, min(k, Samples))
        
        # Temporal filterbank (학습형 1D conv, 입력 형태 (B,1,Chans,Samples))
        self.band_branches = nn.ModuleList()
        for i,(lo,hi) in enumerate(band_defs):
            klen = freq_kernel(lo)
            # 1) Temporal conv (시간축만)
            temp = nn.Conv2d(1, F1, kernel_size=(1, klen), 
                             padding=(0, klen//2), bias=False)
            bn1 = nn.BatchNorm2d(F1)
            # 2) Depthwise spatial (채널 축 결합)
            spatial = nn.Conv2d(F1, F1*D, kernel_size=(Chans,1), 
                                groups=F1, bias=False)
            bn2 = nn.BatchNorm2d(F1*D)
            # 3) 감마 대역은 더 세밀한 주파수/시간 구조 → 추가 짧은 커널 또는 dilation
            if hi >= 60:  # gamma branch heuristic (80 → 60으로 낮춤)
                gamma_extra = nn.Sequential(
                    nn.Conv2d(F1*D, gamma_F1, kernel_size=(1,9), 
                              padding=(0,4), dilation=(1,1), bias=False),
                    nn.BatchNorm2d(gamma_F1),
                    nn.ELU(),
                    nn.Conv2d(gamma_F1, gamma_F1, kernel_size=(1,5), 
                              padding=(0,2), dilation=(1,2), groups=gamma_F1, bias=False),
                    nn.BatchNorm2d(gamma_F1),
                )
            else:
                gamma_extra = nn.Identity()
            
            # Pooling을 약하게: 2x2 = 4배 다운샘플링 (기존 16배에서 줄임)
            branch = nn.Sequential(
                temp, bn1,
                nn.ELU(),
                spatial, bn2,
                nn.ELU(),
                nn.AvgPool2d((1,2)),  # 4 → 2
                nn.Dropout(dropoutRate),
                gamma_extra,
                nn.AvgPool2d((1,2)),  # 4 → 2
                nn.Dropout(dropoutRate),
                # 최종적으로 (1, target_time_dim) 크기로 통일
                nn.AdaptiveAvgPool2d((1, self.target_time_dim))
            )
            self.band_branches.append(branch)

        # Feature channels 계산 (F1=16, D=2 → low:32, mid:32, high:32 → total:96)
        self.band_channels = [ (gamma_F1 if b[1]>=60 else F1*D) for b in band_defs ]
        feature_channels = sum(self.band_channels)
        
        # Dimensionality Reduction Layer (96 → 48)
        self.dim_reduction = nn.Sequential(
            nn.Conv1d(feature_channels, reduced_dim, kernel_size=1),
            nn.BatchNorm1d(reduced_dim),
            nn.ELU()
        )
        
        t_model_dim = reduced_dim  # Use reduced dimension (48)
        
        # Multi-band Attention: 각 주파수 대역별로 독립적인 score 계산
        # 각 밴드의 특징만 보고 해당 밴드의 중요도를 판단 (band-specific)
        self.num_bands = len(band_defs)
        self.band_score = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, max(4, c // 4)),
                nn.ReLU(),
                nn.Linear(max(4, c // 4), 1)
            ) for c in self.band_channels
        ])
        
        # Positional Encoding 추가
        self.pos_enc = PositionalEncoding(t_model_dim, max_len=self.target_time_dim)
        
        # Transformer로 시간 의미 패턴 모델링
        encoder_layer = nn.TransformerEncoderLayer(d_model=t_model_dim, nhead=4, batch_first=True)
        self.seq_model = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 최종 집계 및 분류기 (용량 확대)
        self.classifier = nn.Sequential(
            nn.LayerNorm(t_model_dim),
            nn.Linear(t_model_dim, 64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, nb_classes, bias=False)
        )

    def forward(self, x):
        # x: (B,1,Chans,Samples)
        branch_feats = []
        for b in self.band_branches:
            branch_feats.append(b(x))  # (B, Cb, 1, target_time_dim) - 모두 동일 크기!
        
        # Band-specific Attention: 각 밴드의 global feature를 이용하여 독립적으로 score 계산
        scores = []
        for i, feat in enumerate(branch_feats):
            g = feat.mean(dim=[2, 3])          # (B, Cb) - 밴드별 global feature
            s = self.band_score[i](g)          # (B, 1) - 각 밴드별 독립 MLP로 score 계산
            scores.append(s)
        
        scores = torch.cat(scores, dim=1)      # (B, num_bands)
        band_weights = F.softmax(scores, dim=1)  # (B, num_bands) - softmax로 정규화
        
        # Attention을 각 밴드에 적용
        weighted_branch_feats = []
        for i, feat in enumerate(branch_feats):
            # (B, 1, 1, 1) shape으로 broadcasting
            w = band_weights[:, i].view(-1, 1, 1, 1)
            weighted_feat = feat * w  # (B, Cb, 1, T)
            weighted_branch_feats.append(weighted_feat)
        
        # Weighted features를 concat
        feat = torch.cat(weighted_branch_feats, dim=1)  # (B, sumCb=96, 1, target_time_dim)
        
        # Apply dimensionality reduction (96 → 48)
        feat = feat.squeeze(2)  # (B, 96, target_time_dim)
        feat = self.dim_reduction(feat)  # (B, 48, target_time_dim)
        
        # reshape for sequence model
        feat = feat.transpose(1, 2)  # (B, target_time_dim, 48)
        
        # Positional Encoding 추가
        feat = self.pos_enc(feat)  # (B, target_time_dim, 48)
        
        # sequence modeling with Transformer
        seq_out = self.seq_model(feat)  # (B, target_time_dim, sumCb)
        glob = seq_out.mean(dim=1)      # temporal mean -> (B, sumCb)
        
        logits = self.classifier(glob)
        return logits, glob  # Return both logits and features for center loss
