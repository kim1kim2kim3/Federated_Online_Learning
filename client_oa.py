import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset
from collections import defaultdict
from utils.process_increment import unscaled_metrics
from models.fl_model import GRU
from copy import deepcopy
import numpy as np
import scipy.stats


def compact_history_dataset(dataset):
    """Store delayed-history data without cloning the full source storage.

    `BaseFLServer.update_train_data()` builds one-sample TensorDataset slices
    from the full streaming tensors.  A plain `deepcopy(TensorDataset)` can copy
    the underlying full storage for those views, which is catastrophic in
    full-round REFOL runs.  Clone only the visible slice tensors instead.
    """
    if isinstance(dataset, TensorDataset):
        return TensorDataset(
            *(tensor.detach().clone() for tensor in dataset.tensors)
        )
    return deepcopy(dataset)

class Client(object):

    def __init__(self, client_id, client_dataset, feature_scaler,
                 input_size, output_size, args, model=None):
        self.client_id = client_id
        self.client_dataset = client_dataset
        self.feature_scaler = feature_scaler
        self.input_size = input_size
        self.output_size = output_size
        self.args = args
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size
        self.dataloader = DataLoader(self.client_dataset, batch_size=self.batch_size)

        if model is None:
            self.model = GRU(input_size, self.args.hidden_size, output_size, self.args.dropout, self.args.num_layers)
        else:
            self.model = model

        self.state_dict = None
        self.h_client_dataset = None
        self.selected = False

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')


    def local_execute(self, state_dict_to_load, should_train=False):
        print('Executing on client #{}'.format(self.client_id))

        # REFOL selected/eligible clients must start the round from the latest
        # global model before both current-data evaluation and delayed-label
        # adaptation. Non-selected clients receive None and keep their local
        # model for evaluation-only execution.
        if state_dict_to_load is not None:
            self.model.load_state_dict(state_dict_to_load)

        # 1. EVALUATION (Predict-then-Update):
        # 현재 라운드 학습을 수행하기 전, 기존 모델 가중치로 현재 유입된 데이터(eval_dataset_current)를 먼저 예측 및 평가
        self.model.to(self.device)
        self.model.eval()
        self.dataloader_eval = DataLoader(self.eval_dataset_current, batch_size=self.batch_size)

        epoch_log = defaultdict(lambda: 0.0)
        num_samples = 0

        with torch.no_grad():
            for batch in self.dataloader_eval:
                x, y, x_attr, y_attr = batch
                x = x.to(self.device) if (x is not None) else None
                y = y.to(self.device) if (y is not None) else None
                x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                data = dict(
                    x=x, x_attr=x_attr, y=y, y_attr=y_attr
                )
                y_pred = self.model(data)

                loss_func_name = getattr(self.args, 'loss_func', 'mse')
                if loss_func_name == 'mae':
                    criterion = nn.L1Loss()
                elif loss_func_name == 'smae':
                    criterion = nn.SmoothL1Loss()
                else:
                    criterion = nn.MSELoss()

                loss = criterion(y_pred, y)

                num_samples += x.shape[0]
                metrics = unscaled_metrics(y_pred, y, self.feature_scaler)
                epoch_log['loss'] += loss.detach() * x.shape[0]
                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0]

            for k in epoch_log:
                epoch_log[k] /= num_samples
                epoch_log[k] = epoch_log[k].cpu()

        # 2. ADAPTATION (지연된 라벨 학습):
        # 서버가 명시적으로 학습 대상으로 지정했고, 지연 시간(delay)이 지나 정답 라벨이 도착한 경우에만 학습 진행
        # self.selected는 REFOL의 drift selection 결과 표시용이며 학습 여부를 직접 결정하지 않습니다.
        if should_train and self.train_dataset_delayed is not None:
            self.model.to(self.device)
            self.model.train()

            self.dataloader_train = DataLoader(self.train_dataset_delayed, batch_size=self.batch_size)
            with torch.enable_grad():
                for epoch_i in range(self.args.epoch):
                    for batch in self.dataloader_train:
                        x, y, x_attr, y_attr = batch
                        x = x.to(self.device) if (x is not None) else None
                        y = y.to(self.device) if (y is not None) else None
                        x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                        y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                        data = dict(
                            x=x, x_attr=x_attr, y=y, y_attr=y_attr
                        )
                        y_pred = self.model(data)

                        loss_func_name = getattr(self.args, 'loss_func', 'mse')
                        if loss_func_name == 'mae':
                            criterion = nn.L1Loss()
                        elif loss_func_name == 'smae':
                            criterion = nn.SmoothL1Loss()
                        else:
                            criterion = nn.MSELoss()

                        loss = criterion(y_pred, y)
                        loss.backward()

                        for param in self.model.parameters():
                            param.data = param.data - self.lr * param.grad.data
                            param.grad.data.zero_()

            # 역사적 데이터셋 갱신 (드리프트 비교 기준)
            self.h_client_dataset = compact_history_dataset(self.train_dataset_delayed)

        self.state_dict = deepcopy(self.model.to(self.device).state_dict())

        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)
        self.model.to(self.device)
        self.local_result = {
            'state_dict': self.state_dict, 'log': epoch_log
        }

    def eval_drift(self):
        if getattr(self, 'train_dataset_delayed', None) is None:
            self.selected = False
            return

        if self.h_client_dataset is None:
            self.selected = True
        else:
            self.dataloader = DataLoader(self.train_dataset_delayed, batch_size=self.batch_size)
            for batch in self.dataloader:
                x, y, x_attr, y_attr = batch
                data_now = np.array(x.flatten().tolist())
            self.h_dataloader = DataLoader(self.h_client_dataset, batch_size=self.batch_size)
            for batch in self.h_dataloader:
                x, y, x_attr, y_attr = batch
                data_h = np.array(x.flatten().tolist())
            data_h = self.feature_scaler.inverse_transform(data_h)
            data_now = self.feature_scaler.inverse_transform(data_now)
            KL = scipy.stats.entropy(data_now, data_h)
            if KL > self.args.kl_threshold:
                self.selected = True
            else:
                self.selected = False
