import torch
import json
import random
import gc
import numpy as np
import os
from datetime import datetime
from transformers import AdamW
from config import Config
from models import SCSR, SCRO
from dataset import CustomIterableDataset
from utils import (load_schema, load_data, train_with_early_stopping, test,
                   extract_sroes, SRO, load_model, create_stratified_splits,
                   PerformanceProfiler, save_results_summary, run_statistical_tests)

def set_random_seed(seed):
    if isinstance(seed, str):
        seed = int(seed.split('_')[0])
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_cuda_status():
    print("CUDA状态检查:")
    print(f"  CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA设备数量: {torch.cuda.device_count()}")
        print(f"  当前CUDA设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name()}")
        print(f"  设备内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"  使用设备: {Config.device}")
    print("-" * 40)

def run_single_experiment(train_data, val_data, test_data, run_id, seed, profiler=None):
    print(f"\n  开始实验 {run_id} (seed={seed})")
    set_random_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = Config.device
    try:
        train_data_loader = CustomIterableDataset(train_data, True)
        valid_data_loader = CustomIterableDataset(val_data, False)
        print(f"    初始化模型...")
        scsr = SCSR(Config)
        scro = SCRO()
        scsr.to(device)
        scro.to(device)
        param_optimizer = list(scsr.named_parameters()) + list(scro.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        lr = Config.learning_rate
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        print(f"    开始训练...")
        checkpoint = train_with_early_stopping(train_data_loader, scsr, scro, optimizer, valid_data_loader, profiler)
        scsr.load_state_dict(checkpoint['model4s_state_dict'])
        scro.load_state_dict(checkpoint['model4po_state_dict'])
        model_save_path = f"{Config.PATH_SAVE.replace('.pkl', f'_run_{run_id}.pkl')}"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(checkpoint, model_save_path)
        print(f"    模型已保存到: {model_save_path}")
        print(f"    开始测试...")
        scsr.eval()
        scro.eval()
        if profiler and Config.ENABLE_LATENCY_PROFILING:
            try:
                sample_batch = next(iter(train_data_loader))
                sample_input = [
                    torch.tensor(sample_batch[0][:1], dtype=torch.long).to(device),
                    torch.tensor(sample_batch[1][:1], dtype=torch.long).to(device),
                    torch.tensor(sample_batch[2][:1], dtype=torch.long).to(device),
                    torch.tensor(sample_batch[3][:1], dtype=torch.float).to(device),
                    torch.tensor(sample_batch[4][:1], dtype=torch.int).to(device),
                ]
                avg_latency, std_latency = profiler.profile_inference(scsr, scro, sample_input)
                print(f"    推理延迟: {avg_latency:.2f} ± {std_latency:.2f} ms")
            except Exception as e:
                print(f"    性能分析出错: {e}")
        csv_path = f"{Config.RESULTS_DIR}/detailed_results_run_{run_id}.csv" if hasattr(Config, 'RESULTS_DIR') else None
        try:
            f1, precision, recall = test(test_data, False, scsr, scro, csv_path)
            print(f"    实验 {run_id} 结果: F1={f1:.5f}, Precision={precision:.5f}, Recall={recall:.5f}")
        except Exception as e:
            print(f"    测试阶段出错: {e}")
            f1, precision, recall = 0.0, 0.0, 0.0
        del scsr, scro, optimizer, train_data_loader, valid_data_loader
        torch.cuda.empty_cache()
        gc.collect()
        return {
            'run_id': run_id,
            'seed': seed,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'early_stopped': checkpoint.get('early_stopped', False),
            'best_val_loss': checkpoint.get('best_val_loss', None),
            'model_path': model_save_path
        }
    except Exception as e:
        print(f"    实验 {run_id} 出现严重错误: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return {
            'run_id': run_id,
            'seed': seed,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'early_stopped': False,
            'best_val_loss': None,
            'model_path': None,
            'error': str(e)
        }

def run_cross_validation_experiment(all_data, test_data, run_id, seed, profiler=None):
    print(f"\n开始第 {run_id + 1} 次交叉验证实验 (seed={seed})")
    print("-" * 50)
    set_random_seed(seed)
    try:
        print(f"  创建 {Config.NUM_FOLDS} 折交叉验证分割...")
        splits = create_stratified_splits(all_data, n_splits=Config.NUM_FOLDS, n_repeats=1, random_state=seed)
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits[:Config.NUM_FOLDS]):
            print(f"\n  开始第 {fold_idx + 1} 折 / {Config.NUM_FOLDS} 折...")
            print(f"    训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")
            try:
                train_fold_data = [all_data[i] for i in train_idx if i < len(all_data)]
                val_fold_data = [all_data[i] for i in val_idx if i < len(all_data)]
                print(f"    实际训练样本: {len(train_fold_data)}, 实际验证样本: {len(val_fold_data)}")
                if len(train_fold_data) == 0 or len(val_fold_data) == 0:
                    print(f"    警告: 第 {fold_idx + 1} 折数据为空，跳过")
                    continue
            except IndexError as e:
                print(f"    第 {fold_idx + 1} 折数据索引错误: {e}")
                continue
            fold_result = run_single_experiment(
                train_fold_data, val_fold_data, test_data,
                f"cv{run_id}_fold{fold_idx}", f"{seed}_{fold_idx}", profiler
            )
            fold_result['fold'] = fold_idx
            fold_results.append(fold_result)
        valid_results = [r for r in fold_results if 'error' not in r and r['f1'] > 0]
        if valid_results:
            avg_f1 = np.mean([r['f1'] for r in valid_results])
            avg_precision = np.mean([r['precision'] for r in valid_results])
            avg_recall = np.mean([r['recall'] for r in valid_results])
            print(f"\n第 {run_id + 1} 次交叉验证平均结果 ({len(valid_results)}/{len(fold_results)} 折有效):")
            print(f"  F1: {avg_f1:.5f}, Precision: {avg_precision:.5f}, Recall: {avg_recall:.5f}")
        else:
            avg_f1 = avg_precision = avg_recall = 0.0
            print(f"\n第 {run_id + 1} 次交叉验证失败: 没有有效结果")
        return {
            'run_id': run_id,
            'seed': seed,
            'f1': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall,
            'fold_results': fold_results,
            'num_folds': len(fold_results),
            'valid_folds': len(valid_results)
        }
    except Exception as e:
        print(f"第 {run_id + 1} 次交叉验证出现严重错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'run_id': run_id,
            'seed': seed,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'fold_results': [],
            'num_folds': 0,
            'valid_folds': 0,
            'error': str(e)
        }

def run_cross_validation_only():
    print("开始交叉验证实验...")
    print("=" * 60)
    check_cuda_status()
    if hasattr(Config, 'RESULTS_DIR'):
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    profiler = PerformanceProfiler() if (Config.ENABLE_MEMORY_PROFILING or Config.ENABLE_LATENCY_PROFILING) else None
    device = Config.device
    try:
        print("加载数据和schema...")
        load_schema(Config.PATH_SCHEMA)
        train_path = Config.PATH_TRAIN
        val_path = Config.PATH_VAL
        test_path = Config.PATH_TEST
        train_data = load_data(train_path)
        val_data = load_data(val_path)
        test_data = load_data(test_path)
        all_data = train_data + val_data
        print(f"数据加载完成:")
        print(f"  交叉验证数据: {len(all_data)} 样本 (训练集: {len(train_data)} + 验证集: {len(val_data)})")
        print(f"  测试集: {len(test_data)} 样本")
        print(f"  关系类型数量: {Config.num_p}")
        print(f"  使用设备: {device}")
        print(f"  计划进行 {Config.NUM_REPEATED_SPLITS} 次 {Config.NUM_FOLDS} 折交叉验证")
        all_results = []
        print(f"\n开始 {Config.NUM_REPEATED_SPLITS} 次重复 {Config.NUM_FOLDS} 折交叉验证")
        print("-" * 60)
        for run_id in range(Config.NUM_REPEATED_SPLITS):
            seed = Config.RANDOM_SEEDS[run_id] if run_id < len(Config.RANDOM_SEEDS) else run_id + 42
            cv_result = run_cross_validation_experiment(
                all_data, test_data, run_id, seed, profiler
            )
            cv_result['experiment_type'] = 'cross_validation'
            all_results.append(cv_result)
        efficiency_report = profiler.get_efficiency_report() if profiler else {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"{Config.RESULTS_DIR}/cross_validation_results_{timestamp}.json" if hasattr(Config,
                                                                                                    'RESULTS_DIR') else f"cross_validation_results_{timestamp}.json"
        valid_results = [r for r in all_results if 'error' not in r and r['f1'] > 0]
        if valid_results:
            summary = save_results_summary(valid_results, efficiency_report, results_path)
            print(f"\n交叉验证特定统计:")
            print(f"  总运行次数: {len(all_results)}")
            print(f"  成功运行次数: {len(valid_results)}")
            total_folds = sum([r.get('num_folds', 0) for r in all_results])
            valid_folds = sum([r.get('valid_folds', 0) for r in all_results])
            print(f"  总折数: {total_folds}")
            print(f"  成功折数: {valid_folds}")
            if total_folds > 0:
                print(f"  成功率: {valid_folds / total_folds * 100:.1f}%")
        else:
            print("警告: 没有有效的实验结果用于统计分析")
            summary = {
                'statistical_results': {},
                'efficiency_metrics': efficiency_report,
                'detailed_results': all_results
            }
        print(f"\n交叉验证实验完成！结果已保存到: {results_path}")
        return summary
    except Exception as e:
        print(f"交叉验证实验过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_simple_train():
    print("开始简单训练模式...")
    check_cuda_status()
    device = Config.device
    try:
        load_schema(Config.PATH_SCHEMA)
        train_path = Config.PATH_TRAIN
        val_path = Config.PATH_VAL
        test_path = Config.PATH_TEST
        all_data = load_data(train_path)
        valid_data = load_data(val_path)
        test_data = load_data(test_path)
        random.shuffle(all_data)
        train_data = all_data
        print(f"数据加载完成:")
        print(f"  训练集: {len(train_data)} 样本")
        print(f"  验证集: {len(valid_data)} 样本")
        print(f"  测试集: {len(test_data)} 样本")
        train_data_loader = CustomIterableDataset(train_data, True)
        valid_data_loader = CustomIterableDataset(valid_data, False)
        scsr = SCSR(Config)
        scro = SCRO()
        scsr.to(device)
        scro.to(device)
        print(f"SCSR参数数量: {sum(p.numel() for p in scsr.parameters() if p.requires_grad)}")
        print(f"SCRO参数数量: {sum(p.numel() for p in scro.parameters() if p.requires_grad)}")
        param_optimizer = list(scsr.named_parameters()) + list(scro.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        lr = Config.learning_rate
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        checkpoint = train_with_early_stopping(train_data_loader, scsr, scro, optimizer, valid_data_loader)
        del train_data
        gc.collect()
        model_path = Config.PATH_SAVE
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(checkpoint, model_path)
        print('模型已保存!')
        scsr.load_state_dict(checkpoint['model4s_state_dict'])
        scro.load_state_dict(checkpoint['model4po_state_dict'])
        scsr.to(device)
        scro.to(device)
        scsr.eval()
        scro.eval()
        f1, precision, recall = test(test_data, True, scsr, scro)
        print('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
    except Exception as e:
        print(f"简单训练模式出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='运行关系抽取实验')
    parser.add_argument('--mode', type=str, default='cv',
                        choices=['simple', 'cv'],
                        help='运行模式: simple(单次训练) 或 cv(交叉验证)')
    args = parser.parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if args.mode == 'cv':
        run_cross_validation_only()
    else:
        run_simple_train()
