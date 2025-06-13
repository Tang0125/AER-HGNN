import numpy as np
import torch
from torch.utils.data import IterableDataset
from config import Config
from itertools import cycle


class CustomIterableDataset(IterableDataset):
    def __init__(self, data, random):
        super(CustomIterableDataset, self).__init__()
        self.data = data
        self.random = random
        self.tokenizer = Config.tokenizer

    def __len__(self):
        return len(self.data)

    def search(self, sequence, pattern):
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def process_data(self):
        idxs = list(range(len(self.data)))
        if self.random:
            np.random.shuffle(idxs)
        batch_size = Config.batch_size
        max_seq_len = Config.max_seq_len
        num_p = Config.num_p
        batch_token_ids = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        batch_mask_ids = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        batch_segment_ids = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        batch_subject_ids = np.zeros((batch_size, 2), dtype=np.int32)
        batch_subject_labels = np.zeros((batch_size, max_seq_len, 2), dtype=np.int32)
        batch_object_labels = np.zeros((batch_size, max_seq_len, num_p, 2), dtype=np.int32)
        batch_i = 0

        for i in idxs:
            try:
                text = self.data[i]['text']
                batch_token_ids[batch_i, :] = self.tokenizer.encode(text, max_length=max_seq_len,
                                                                    pad_to_max_length=True,
                                                                    add_special_tokens=True)
                batch_mask_ids[batch_i, :len(text) + 2] = 1
                sro_list = self.data[i]['sro_list']

                if len(sro_list) == 0:

                    continue

                idx = np.random.randint(0, len(sro_list), size=1)[0]
                s_rand = self.tokenizer.encode(sro_list[idx][0])[1:-1]
                s_rand_idx = self.search(list(batch_token_ids[batch_i, :]), s_rand)

                if s_rand_idx == -1:

                    continue

                batch_subject_ids[batch_i, :] = [s_rand_idx, s_rand_idx + len(s_rand) - 1]

                for j in range(len(sro_list)):
                    sro = sro_list[j]
                    s = self.tokenizer.encode(sro[0])[1:-1]


                    if sro[1] not in Config.predicate2id:
                        continue

                    p = Config.predicate2id[sro[1]]
                    o = self.tokenizer.encode(sro[2])[1:-1]
                    s_idx = self.search(list(batch_token_ids[batch_i]), s)
                    o_idx = self.search(list(batch_token_ids[batch_i]), o)

                    if s_idx != -1 and o_idx != -1:
                        batch_subject_labels[batch_i, s_idx, 0] = 1
                        batch_subject_labels[batch_i, s_idx + len(s) - 1, 1] = 1
                        if s_idx == s_rand_idx:
                            batch_object_labels[batch_i, o_idx, p, 0] = 1
                            batch_object_labels[batch_i, o_idx + len(o) - 1, p, 1] = 1

                batch_i += 1

            except Exception as e:

                print(f"处理样本 {i} 时出错: {e}")
                continue

            if batch_i == batch_size or i == idxs[-1]:
                if batch_i > 0:
                    yield batch_token_ids, batch_mask_ids, batch_segment_ids, batch_subject_labels, batch_subject_ids, batch_object_labels
                batch_token_ids[:, :] = 0
                batch_mask_ids[:, :] = 0
                batch_subject_ids[:, :] = 0
                batch_subject_labels[:, :, :] = 0
                batch_object_labels[:, :, :, :] = 0
                batch_i = 0

    def get_stream(self):
        return cycle(self.process_data())

    def __iter__(self):
        return self.get_stream()