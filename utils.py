import json
import numpy as np
import torch
import torch.nn as nn
import time
import re
import os
import psutil
import gc
import random
from transformers import AdamW
from config import Config
from models import SCSR, SCRO
from dataset import CustomIterableDataset
from torch.utils.tensorboard import SummaryWriter
import csv
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

device = Config.device

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SRO(tuple):
    def __init__(self, sro):
        self.srox = (
            tuple(token.lower().replace(" ", "") for token in Config.tokenizer.tokenize(sro[0])),
            sro[1].strip().lower().replace(" ", ""),
            tuple(token.lower().replace(" ", "") for token in Config.tokenizer.tokenize(sro[2])),
        )

    def __hash__(self):
        return hash(self.srox)

    def __eq__(self, sro):
        return self.srox == sro.srox

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

class PerformanceProfiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.inference_latencies = []
        self.training_latencies = []

    def get_memory_usage(self):
        cpu_memory = psutil.virtual_memory().used / (1024 ** 3)
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        return cpu_memory, gpu_memory

    def profile_inference(self, model1, model2, sample_input):
        model1.eval()
        model2.eval()
        latencies = []
        for _ in range(Config.WARMUP_STEPS):
            with torch.no_grad():
                _ = model1(*sample_input[:3])
        for _ in range(50):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            with torch.no_grad():
                subject_pred, hidden_states = model1(*sample_input[:3])
                object_pred = model2(hidden_states, sample_input[4], sample_input[1])
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
        self.inference_latencies.extend(latencies)
        return np.mean(latencies), np.std(latencies)

    def record_memory(self):
        cpu_mem, gpu_mem = self.get_memory_usage()
        self.cpu_memory_usage.append(cpu_mem)
        self.gpu_memory_usage.append(gpu_mem)

    def get_efficiency_report(self):
        report = {
            'avg_inference_latency_ms': np.mean(self.inference_latencies) if self.inference_latencies else 0,
            'std_inference_latency_ms': np.std(self.inference_latencies) if self.inference_latencies else 0,
            'max_cpu_memory_gb': max(self.cpu_memory_usage) if self.cpu_memory_usage else 0,
            'max_gpu_memory_gb': max(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'avg_cpu_memory_gb': np.mean(self.cpu_memory_usage) if self.cpu_memory_usage else 0,
            'avg_gpu_memory_gb': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
        }
        return report

def load_model():
    load_schema(Config.PATH_SCHEMA)
    checkpoint = torch.load(Config.PATH_MODEL, map_location=device)
    scsr = SCSR(Config)
    scsr.load_state_dict(checkpoint['model4s_state_dict'])
    scsr.to(device)
    scro = SCRO()
    scro.load_state_dict(checkpoint['model4po_state_dict'])
    scro.to(device)
    return scsr, scro

def extract_sroes(text, model4s, model4po):
    with torch.no_grad():
        tokenizer = Config.tokenizer
        max_seq_len = Config.max_seq_len
        token_ids = torch.tensor(
            tokenizer.encode(text, max_length=max_seq_len, pad_to_max_length=True, add_special_tokens=True)).view(1, -1).to(device)
        if len(text) > max_seq_len - 2:
            text = text[:max_seq_len - 2]
        mask_ids = torch.tensor([1] * (len(text) + 2) + [0] * (max_seq_len - len(text) - 2)).view(1, -1).to(device)
        segment_ids = torch.tensor([0] * max_seq_len).view(1, -1).to(device)
        subject_labels_pred, hidden_states = model4s(token_ids, mask_ids, segment_ids)
        subject_labels_pred = subject_labels_pred.cpu()
        subject_labels_pred[0, len(text) + 2:, :] = 0
        start = np.where(subject_labels_pred[0, :, 0] > 0.4)[0]
        end = np.where(subject_labels_pred[0, :, 1] > 0.4)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        if len(subjects) == 0:
            return []
        subject_ids = torch.tensor(subjects).view(1, -1).to(device)
        sroes = []
        for s in subjects:
            object_labels_pred = model4po(hidden_states, subject_ids, mask_ids)
            object_labels_pred = object_labels_pred.view((1, max_seq_len, Config.num_p, 2)).cpu()
            object_labels_pred[0, len(text) + 2:, :, :] = 0
            start = np.where(object_labels_pred[0, :, :, 0] > 0.4)
            end = np.where(object_labels_pred[0, :, :, 1] > 0.4)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        sroes.append((s, predicate1, (_start, _end)))
                        break
        id_str = ['[CLS]']
        i = 1
        index = 0
        while i < token_ids.shape[1]:
            if token_ids[0][i] == 102:
                break
            word = tokenizer.decode(token_ids[0, i:i + 1])
            word = re.sub('#+', '', word)
            if word != '[UNK]':
                id_str.append(word)
                index += len(word)
                i += 1
            else:
                j = i + 1
                while j < token_ids.shape[1]:
                    if token_ids[0][j] == 102:
                        break
                    word_j = tokenizer.decode(token_ids[0, j:j + 1])
                    if word_j != '[UNK]':
                        break
                    j += 1
                if token_ids[0][j] == 102 or j == token_ids.shape[1]:
                    while i < j - 1:
                        id_str.append('')
                        i += 1
                    id_str.append(text[index:])
                    i += 1
                    break
                else:
                    index_end = text[index:].find(word_j)
                    word = text[index:index + index_end]
                    id_str.append(word)
                    index += index_end
                    i += 1
        res = []
        for s, r, o in sroes:
            s_start = s[0]
            s_end = s[1]
            sub = ''.join(id_str[s_start:s_end + 1])
            o_start = o[0]
            o_end = o[1]
            obj = ''.join(id_str[o_start:o_end + 1])
            res.append((sub, Config.id2predicate[r], obj))
        return res

def load_schema(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
        predicate = list(data.keys())
        prediction2id = {}
        id2predicate = {}
        for i in range(len(predicate)):
            prediction2id[predicate[i]] = i
            id2predicate[i] = predicate[i]
    num_p = len(predicate)
    Config.predicate2id = prediction2id
    Config.id2predicate = id2predicate
    Config.num_p = num_p

def load_data(path):
    text_sros = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
        for item in data:
            text = item['text']
            sro_list = item['sro_list']
            text_sros.append({
                'text': text,
                'sro_list': sro_list
            })
    return text_sros

def check_tensor_validity(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"警告: {name} 包含 NaN 值")
        return False
    if torch.isinf(tensor).any():
        print(f"警告: {name} 包含无穷值")
        return False
    if (tensor < 0).any() or (tensor > 1).any():
        print(f"警告: {name} 的值超出 [0,1] 范围")
        return False
    return True

def safe_clamp(tensor, min_val=1e-7, max_val=1 - 1e-7):
    return torch.clamp(tensor, min=min_val, max=max_val)

def loss_fn(pred, target):
    if not check_tensor_validity(pred, "predictions"):
        print(f"预测值范围: min={pred.min():.6f}, max={pred.max():.6f}")
        pred = safe_clamp(pred)
    if not check_tensor_validity(target, "targets"):
        print(f"目标值范围: min={target.min():.6f}, max={target.max():.6f}")
        target = safe_clamp(target)
    loss_fct = nn.BCELoss(reduction='none')
    return loss_fct(pred, target)

def create_stratified_splits(data, n_splits=5, n_repeats=3, random_state=42):
    print(f"  正在创建分层分割: {n_splits}折, {n_repeats}次重复, 随机种子={random_state}")
    print(f"  数据样本数: {len(data)}")
    labels = []
    for i, item in enumerate(data):
        try:
            label_set = set()
            sro_list = item.get('sro_list', [])
            if not sro_list:
                label_set.add('NO_RELATION')
            else:
                for sro in sro_list:
                    if len(sro) >= 2:
                        label_set.add(sro[1])
            if label_set:
                label_hash = hash(frozenset(label_set)) % 1000
            else:
                label_hash = 0
            labels.append(label_hash)
        except Exception as e:
            print(f"  警告: 处理样本 {i} 时出错: {e}")
            labels.append(0)
    print(f"  创建的标签数: {len(labels)}")
    print(f"  唯一标签数: {len(set(labels))}")
    unique_labels = set(labels)
    if len(unique_labels) < n_splits:
        print(f"  警告: 唯一标签数 ({len(unique_labels)}) 少于折数 ({n_splits})")
        print(f"  将使用简单随机分割而不是分层分割")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []
        for _ in range(n_repeats):
            splits.extend(list(kf.split(data)))
        return splits
    from sklearn.model_selection import RepeatedStratifiedKFold
    try:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        splits = list(rskf.split(data, labels))
        print(f"  成功创建 {len(splits)} 个分割")
        for i, (train_idx, val_idx) in enumerate(splits):
            if len(train_idx) == 0 or len(val_idx) == 0:
                print(f"  警告: 分割 {i} 包含空的训练或验证集")
            if max(max(train_idx), max(val_idx)) >= len(data):
                print(f"  警告: 分割 {i} 包含超出数据范围的索引")
        return splits
    except Exception as e:
        print(f"  分层分割失败: {e}")
        print(f"  回退到简单随机分割")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []
        for _ in range(n_repeats):
            splits.extend(list(kf.split(data)))
        return splits

def train_with_early_stopping(train_data_loader, scsr, scro, optimizer, valid_data_loader, profiler=None):
    writer = SummaryWriter(log_dir='./runs/experiment')
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        min_delta=Config.EARLY_STOPPING_MIN_DELTA,
        mode='min'
    )
    steps_per_epoch = len(train_data_loader) // Config.batch_size
    print(f"Steps per epoch: {steps_per_epoch}")
    global_step = 0
    best_model_state = None
    scsr.to(device)
    scro.to(device)
    for epoch in range(Config.EPOCH):
        begin_time = time.time()
        scsr.train()
        scro.train()
        train_loss = 0.
        if profiler and Config.ENABLE_MEMORY_PROFILING:
            profiler.record_memory()
        for bi, batch in enumerate(train_data_loader):
            if bi >= steps_per_epoch:
                break
            batch_token_ids, batch_mask_ids, batch_segment_ids, batch_subject_labels, batch_subject_ids, batch_object_labels = batch
            batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(device)
            batch_mask_ids = torch.tensor(batch_mask_ids, dtype=torch.long).to(device)
            batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long).to(device)
            batch_subject_labels = torch.tensor(batch_subject_labels, dtype=torch.float).to(device)
            batch_object_labels = torch.tensor(batch_object_labels, dtype=torch.float).view(Config.batch_size, Config.max_seq_len, Config.num_p * 2).to(device)
            batch_subject_ids = torch.tensor(batch_subject_ids, dtype=torch.int).to(device)
            try:
                batch_subject_labels_pred, hidden_states = scsr(batch_token_ids, batch_mask_ids, batch_segment_ids)
                batch_subject_labels_pred = safe_clamp(batch_subject_labels_pred)
                loss4s = loss_fn(batch_subject_labels_pred, batch_subject_labels.to(torch.float32))
                loss4s = torch.mean(loss4s, dim=2, keepdim=False) * batch_mask_ids
                loss4s = torch.sum(loss4s)
                loss4s = loss4s / torch.sum(batch_mask_ids)
                batch_object_labels_pred = scro(hidden_states, batch_subject_ids, batch_mask_ids)
                batch_object_labels_pred = safe_clamp(batch_object_labels_pred)
                loss4po = loss_fn(batch_object_labels_pred, batch_object_labels.to(torch.float32))
                loss4po = torch.mean(loss4po, dim=2, keepdim=False) * batch_mask_ids
                loss4po = torch.sum(loss4po)
                loss4po = loss4po / torch.sum(batch_mask_ids)
                loss = loss4s + loss4po
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 第 {bi} 批次的损失为 {loss}, 跳过此批次")
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(scsr.parameters()) + list(scro.parameters()), max_norm=1.0)
                optimizer.step()
                train_loss += float(loss.item())
                global_step += 1
                writer.add_scalar('Loss/Train', float(loss.item()), global_step)
                print(f'batch: {bi}, loss: {float(loss.item())}')
            except RuntimeError as e:
                print(f"批次 {bi} 出现运行时错误: {e}")
                continue
            if global_step % Config.VALIDATION_FREQUENCY == 0 and valid_data_loader is not None:
                scsr.eval()
                scro.eval()
                val_loss = evaluate(valid_data_loader, scsr, scro)
                print(f'Step {global_step} - 验证损失: {val_loss:.5f}')
                writer.add_scalar('Loss/Validation', val_loss, global_step)
                if early_stopping(val_loss):
                    print(f'Early stopping triggered at step {global_step}')
                    writer.close()
                    return {
                        "model4s_state_dict": best_model_state['scsr'] if best_model_state else scsr.state_dict(),
                        "model4po_state_dict": best_model_state['scro'] if best_model_state else scro.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "early_stopped": True,
                        "best_val_loss": early_stopping.best_score
                    }
                if val_loss == early_stopping.best_score:
                    best_model_state = {
                        'scsr': scsr.state_dict().copy(),
                        'scro': scro.state_dict().copy()
                    }
                scsr.train()
                scro.train()
        print(f'Epoch {epoch + 1}/{Config.EPOCH} - Training loss: {train_loss / max(1, steps_per_epoch)} - Time: {time.time() - begin_time:.2f}s')
    writer.close()
    return {
        "model4s_state_dict": best_model_state['scsr'] if best_model_state else scsr.state_dict(),
        "model4po_state_dict": best_model_state['scro'] if best_model_state else scro.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "early_stopped": False,
        "best_val_loss": early_stopping.best_score if early_stopping.best_score else train_loss / max(1, steps_per_epoch)
    }

def normalize_sro(sro):
    return tuple(s.strip().lower().replace(" ", "") for s in sro)

def partial_match(pred_set, gold_set):
    pred = {(str(i[0]).split(' ')[0] if isinstance(i[0], str) and len(i[0].split(' ')) > 0 else str(i[0]),
             i[1],
             str(i[2]).split(' ')[0] if isinstance(i[2], str) and len(i[2].split(' ')) > 0 else str(i[2]))
            for i in pred_set}
    gold = {(str(i[0]).split(' ')[0] if isinstance(i[0], str) and len(i[0].split(' ')) > 0 else str(i[0]),
             i[1],
             str(i[2]).split(' ')[0] if isinstance(i[2], str) and len(i[2].split(' ')) > 0 else str(i[2]))
            for i in gold_set}
    return pred, gold

def test(data, is_print, scsr, scro, csv_file_path=None):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    if csv_file_path:
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['text', 'R (Model Extracted)', 'T (True SROs)'])
    for d in data:
        R = set([normalize_sro(SRO(sro)) for sro in extract_sroes(d['text'], scsr, scro)])
        T = set([normalize_sro(SRO(sro)) for sro in d['sro_list']])
        R, T = partial_match(R, T)
        if is_print:
            print('text:', d['text'])
            print('R:', R)
            print('T:', T)
            print(f'Length of R: {len(R)}, Length of T: {len(T)}')
        if csv_file_path:
            csv_writer.writerow([d['text'], list(R), list(T)])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    if csv_file_path:
        csv_file.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

def evaluate(valid_data_loader, scsr, scro):
    scsr.eval()
    scro.eval()
    scsr.to(device)
    scro.to(device)
    val_loss = 0
    steps_per_epoch = len(valid_data_loader) // Config.batch_size
    valid_batches = 0
    with torch.no_grad():
        for bi, batch in enumerate(valid_data_loader):
            if bi >= steps_per_epoch:
                break
            batch_token_ids, batch_mask_ids, batch_segment_ids, batch_subject_labels, batch_subject_ids, batch_object_labels = batch
            batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(device)
            batch_mask_ids = torch.tensor(batch_mask_ids, dtype=torch.long).to(device)
            batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long).to(device)
            batch_subject_labels = torch.tensor(batch_subject_labels, dtype=torch.float).to(device)
            batch_object_labels = torch.tensor(batch_object_labels, dtype=torch.float).view(Config.batch_size, Config.max_seq_len, Config.num_p * 2).to(device)
            batch_subject_ids = torch.tensor(batch_subject_ids, dtype=torch.int).to(device)
            try:
                batch_subject_labels_pred, hidden_states = scsr(batch_token_ids, batch_mask_ids, batch_segment_ids)
                batch_subject_labels_pred = safe_clamp(batch_subject_labels_pred)
                loss4s = loss_fn(batch_subject_labels_pred, batch_subject_labels.to(torch.float32))
                loss4s = torch.mean(loss4s, dim=2) * batch_mask_ids
                loss4s = torch.sum(loss4s) / torch.sum(batch_mask_ids)
                batch_object_labels_pred = scro(hidden_states, batch_subject_ids, batch_mask_ids)
                batch_object_labels_pred = safe_clamp(batch_object_labels_pred)
                loss4po = loss_fn(batch_object_labels_pred, batch_object_labels.to(torch.float32))
                loss4po = torch.mean(loss4po, dim=2) * batch_mask_ids
                loss4po = torch.sum(loss4po) / torch.sum(batch_mask_ids)
                loss = loss4s + loss4po
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += float(loss.item())
                    valid_batches += 1
            except RuntimeError as e:
                print(f"验证批次 {bi} 出现错误: {e}")
                continue
    return val_loss / max(1, valid_batches)

def run_statistical_tests(results_list):
    f1_scores = [r['f1'] for r in results_list]
    precision_scores = [r['precision'] for r in results_list]
    recall_scores = [r['recall'] for r in results_list]
    stats_results = {
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'num_runs': len(results_list)
    }
    confidence_level = 0.95
    alpha = 1 - confidence_level
    for metric in ['f1', 'precision', 'recall']:
        scores = [r[metric] for r in results_list]
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        n = len(scores)
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin_error = t_critical * (std_score / np.sqrt(n))
        stats_results[f'{metric}_ci_lower'] = mean_score - margin_error
        stats_results[f'{metric}_ci_upper'] = mean_score + margin_error
    return stats_results

def save_results_summary(all_results, efficiency_report, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats_results = run_statistical_tests(all_results)
    summary = {
        'statistical_results': stats_results,
        'efficiency_metrics': efficiency_report,
        'detailed_results': all_results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 50)
    print("实验结果摘要")
    print("=" * 50)
    print(f"F1 Score: {stats_results['f1_mean']:.4f} ± {stats_results['f1_std']:.4f}")
    print(f"F1 95% CI: [{stats_results['f1_ci_lower']:.4f}, {stats_results['f1_ci_upper']:.4f}]")
    print(f"Precision: {stats_results['precision_mean']:.4f} ± {stats_results['precision_std']:.4f}")
    print(f"Recall: {stats_results['recall_mean']:.4f} ± {stats_results['recall_std']:.4f}")
    print(f"运行次数: {stats_results['num_runs']}")
    print("\n计算效率指标:")
    print(f"平均推理延迟: {efficiency_report['avg_inference_latency_ms']:.2f} ± {efficiency_report['std_inference_latency_ms']:.2f} ms")
    print(f"最大GPU内存使用: {efficiency_report['max_gpu_memory_gb']:.2f} GB")
    print(f"最大CPU内存使用: {efficiency_report['max_cpu_memory_gb']:.2f} GB")
    print("=" * 50)
    return summary
