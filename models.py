import torch
import torch.nn as nn
from transformers import BertModel
from config import Config
import torch.nn.functional as F

device = Config.device

class SCSR(nn.Module):
    def __init__(self, config, hidden_size=768):
        super(SCSR, self).__init__()
        self.bert = BertModel.from_pretrained(Config.PATH_BERT).to(device)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=hidden_size, out_features=2, bias=True).to(device)
        self.sigmoid = nn.Sigmoid()
        self.gat = HGAT(config).to(device)

    def forward(self, input_ids, input_mask, segment_ids, hidden_size=768):
        hidden_states = self.bert(input_ids,
                                  attention_mask=input_mask,
                                  token_type_ids=segment_ids)[0]
        x, p = self.gat(hidden_states)
        output = self.sigmoid(self.linear(self.dropout(x)))
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        return output, x

class SCRO(nn.Module):
    def __init__(self, num_p=Config.num_p, hidden_size=768):
        super(SCRO, self).__init__()
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_p * 2, bias=True).to(device)
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Linear(hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states, batch_subject_ids, input_mask):
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            print("警告: hidden_states 包含无效值")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        all_s = torch.zeros((hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]),
                            dtype=torch.float32, device=device)
        for b in range(hidden_states.shape[0]):
            try:
                s_start = batch_subject_ids[b][0].item() if isinstance(batch_subject_ids[b][0], torch.Tensor) else \
                batch_subject_ids[b][0]
                s_end = batch_subject_ids[b][1].item() if isinstance(batch_subject_ids[b][1], torch.Tensor) else \
                batch_subject_ids[b][1]
                s_start = max(0, min(s_start, hidden_states.shape[1] - 1))
                s_end = max(0, min(s_end, hidden_states.shape[1] - 1))
                s = hidden_states[b][s_start] + hidden_states[b][s_end]
                cue_len = torch.sum(input_mask[b]).item()
                cue_len = max(1, min(cue_len, hidden_states.shape[1]))
                attn_weights = F.softmax(self.attention(hidden_states[b, :cue_len, :]), dim=0)
                s_x = torch.sum(attn_weights * hidden_states[b, :cue_len, :], dim=0)
                for i in range(cue_len):
                    combined = torch.cat([s_x, hidden_states[b, i]], dim=-1)
                    gate = torch.sigmoid(self.gate(combined))
                    fused = gate * s_x + (1 - gate) * hidden_states[b, i]
                    all_s[b, i, :] = fused
            except Exception as e:
                print(f"处理批次 {b} 时出错: {e}")
                all_s[b] = hidden_states[b]
        hidden_states = hidden_states + all_s
        output = self.sigmoid(self.linear(self.dropout(hidden_states)))
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        return output

class HGAT(nn.Module):
    def __init__(self, config):
        super(HGAT, self).__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.embeding = nn.Embedding(config.num_p, hidden_size).to(device)
        self.relation = nn.Linear(hidden_size, hidden_size).to(device)
        self.down = nn.Linear(3 * hidden_size, hidden_size).to(device)
        self.start_head = nn.Linear(hidden_size, 1).to(device)
        self.end_head = nn.Linear(hidden_size, 1).to(device)
        self.start_tail = nn.Linear(hidden_size, 1).to(device)
        self.end_tail = nn.Linear(hidden_size, 1).to(device)
        self.layers = nn.ModuleList(
            [GATLayer(hidden_size).to(device) for _ in range(config.gat_layers)])

    def forward(self, x, sub_head=None, sub_tail=None, mask=None):
        p = torch.arange(self.config.num_p).long().to(device)
        p = self.relation(self.embeding(p))
        p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))
        x, p = self.gat_layer(x, p, mask)
        return x, p

    def gat_layer(self, x, p, mask=None):
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p

class GATLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ra1 = RelationAttention(hidden_size).to(device)
        self.ra2 = RelationAttention(hidden_size).to(device)

    def forward(self, x, p, mask=None):
        x_ = self.ra1(x, p)
        x = x_ + x
        p_ = self.ra2(p, x, mask)
        p = p_ + p
        return x, p

class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size).to(device)
        self.key = nn.Linear(hidden_size, hidden_size).to(device)
        self.value = nn.Linear(hidden_size, hidden_size).to(device)
        self.score = nn.Linear(2 * hidden_size, 1).to(device)
        self.gate = nn.Linear(hidden_size * 2, 1).to(device)

    def forward(self, p, x, mask=None):
        q = self.query(p)
        k = self.key(x)
        score = self.fuse(q, k)
        if mask is not None:
            mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
            score = score.masked_fill(mask == 1, -1e9)
        score = F.softmax(score, 2)
        v = self.value(x)
        out = torch.einsum('bcl,bld->bcd', score, v) + p
        g = self.gate(torch.cat([out, p], 2)).sigmoid()
        out = g * out + (1 - g) * p
        return out

    def fuse(self, x, y):
        x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        temp = torch.cat([x, y], 3)
        return self.score(temp).squeeze(3)
