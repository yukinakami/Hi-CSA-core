import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 root_path: str = 'data',
                 flag: str = 'train',
                 seq_len: int = 96,
                 pred_len: int = 24,
                 target_col: str = 'OT',
                 features: str = 'M',
                 scale: bool = True,
                 scaler=None,
                 cols: list = None,
                 split_strategy: str = 'ratio'
                 ):
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag
        self.scale = scale
        self.features = features
        self.target_col = target_col
        self.split_strategy = split_strategy

        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        self.data_len = len(df_raw)

        if split_strategy == 'ett':
            file_name = os.path.basename(data_path).lower()
            if file_name.startswith('ettm'):
                num_train = 12 * 30 * 24 * 4
                num_vali = 4 * 30 * 24 * 4
                num_test = 4 * 30 * 24 * 4
            elif file_name.startswith('etth'):
                num_train = 12 * 30 * 24
                num_vali = 4 * 30 * 24
                num_test = 4 * 30 * 24
            else:
                raise ValueError("split_strategy='ett' only supports ETTh*/ETTm* data files")

            border1s = [
                0,
                num_train - self.seq_len,
                num_train + num_vali - self.seq_len
            ]
            border2s = [
                num_train,
                num_train + num_vali,
                num_train + num_vali + num_test
            ]
            border2s = [min(border2, len(df_raw)) for border2 in border2s]
        elif split_strategy in ('ratio', 'standard'):
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test

            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
        else:
            raise ValueError("split_strategy must be 'ratio', 'standard', or 'ett'")

        border1s = [max(0, border1) for border1 in border1s]
        
        border1 = border1s[['train', 'val', 'test'].index(flag)]
        border2 = border2s[['train', 'val', 'test'].index(flag)]
        self.border1 = border1
        self.border2 = border2
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target_col]]
            
        data = df_data.values
        
        if self.scale:
            if flag == 'train':
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)
            else:
                if scaler is None:
                    raise ValueError("Scaler must be provided for val/test sets when scale=True")
                self.scaler = scaler
                data = self.scaler.transform(data)

        self.global_train_data = data[border1s[0]:border2s[0]].astype(np.float32)
        
        self.data_x = data[border1:border2].astype(np.float32)
        self.data_y = data[border1:border2].astype(np.float32)

        if self.features == 'MS':
            df_target = df_raw[[self.target_col]]
            target_data = df_target.values
            if self.scale:
                pass
            self.data_y = target_data[border1:border2].astype(np.float32)

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):

        s_begin_micro = index
        s_end_micro = s_begin_micro + self.seq_len

        r_begin = s_end_micro
        r_end = r_begin + self.pred_len

        seq_x_micro = self.data_x[s_begin_micro : s_end_micro]
        seq_y = self.data_y[r_begin : r_end]

        assert seq_x_micro.shape[0] == self.seq_len, f"Micro len error: {seq_x_micro.shape[0]} vs {self.seq_len}"
        assert seq_y.shape[0] == self.pred_len, f"Pred len error: {seq_y.shape[0]} vs {self.pred_len}"

        return (
            torch.from_numpy(seq_x_micro.copy()),
            torch.from_numpy(seq_y.copy()),
        )

def get_dataloader(root_path, data_path, flag, batch_size=32, seq_len=96, pred_len=24, num_workers=4, split_strategy='ratio'):
    dataset = TimeSeriesDataset(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        seq_len=seq_len,
        pred_len=pred_len,
        split_strategy=split_strategy
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(flag=='train'),
        num_workers=num_workers,
        drop_last=True
    )
    return loader
