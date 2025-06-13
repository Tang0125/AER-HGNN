import json
from transformers import BertTokenizer
import torch


class Config:
    batch_size = 16
    max_seq_len = 128
    num_p = 23
    learning_rate = 1e-5
    EPOCH = 10
    hidden_size = 768
    gat_layers = 2


    NUM_RUNS = 5  # 进行5次独立运行
    NUM_FOLDS = 3  # 3折交叉验证
    NUM_REPEATED_SPLITS = 3  # 重复分层随机分割次数

    ENABLE_CROSS_VALIDATION = True

    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 1e-3

    VALIDATION_FREQUENCY = 200


    # 计算效率量化配置
    ENABLE_MEMORY_PROFILING = True
    ENABLE_LATENCY_PROFILING = True
    WARMUP_STEPS = 10

    PATH_SCHEMA = "schema.json"
    PATH_TRAIN = 'data/train.json'
    PATH_VAL = 'data/val.json'
    PATH_TEST = 'data/test.json'
    PATH_BERT = "model/bert-base-chinese"
    PATH_MODEL = "model/apsc_re/apsc_re.pkl"
    PATH_SAVE = "model/apsc_re/apsc_re.pkl"


    RESULTS_DIR = "results/"

    tokenizer = BertTokenizer.from_pretrained("model/apsc_re/vocab.txt")

    id2predicate = {}
    predicate2id = {}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    RANDOM_SEEDS = [42, 123, 456, 789, 1000]