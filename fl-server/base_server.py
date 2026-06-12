import numpy as np
import torch
import xlsxwriter
from torch.utils.data import TensorDataset
from copy import deepcopy
from utils.process_increment import load_dataset

class BaseFLServer(object):
    """
    연합 학습(FL) 서버의 공통 인터페이스 및 데이터 처리 보일러플레이트를 정의한 베이스 클래스입니다.
    새로운 FL 알고리즘(예: FedAvg, FedProx)을 추가할 때 이 클래스를 상속받아 필요한 메서드만 오버라이드하면 됩니다.
    """
    def __init__(self, config):
        self.config = config
        self.global_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def boot(self):
        """
        데이터셋을 로드하고 클라이언트 모델들을 생성하는 공통 초기화 로직입니다.
        """
        from client_oa import Client

        print('Booting {} fl-server...'.format(self.config.agg_model))
        self.num_clients = self.config.num_clients
        print('Total clients: {}'.format(self.num_clients))
        
        # 1. 데이터셋 로드
        data, selected_node = load_dataset(
            name=self.config.dataset,
            adj_mx_name=self.config.adj_mx,
            num_clients=self.num_clients,
            pred_len=self.config.pred_steps
        )
        self.data = data
        input_size = self.data['x'].shape[-1] + self.data['x_attr'].shape[-1]
        output_size = self.data['y'].shape[-1]
        self.max_epoch = data['x'].shape[0]
        self.train_per_num_samples = 1

        # 2. 난수 시드 설정
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # 3. 클라이언트 인스턴스 초기 배치
        clients = []
        for client_i in range(self.num_clients):
            client_dataset = TensorDataset(
                data['x'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['y'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['x_attr'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['y_attr'][:self.train_per_num_samples, :, client_i:client_i + 1, :]
            )

            client_tmp = Client(
                client_id=client_i,
                client_dataset=client_dataset,
                feature_scaler=self.data['feature_scaler'],
                input_size=input_size,
                output_size=output_size,
                args=self.config
            )
            clients.append(client_tmp)

        self.clients = clients

        self.server_datasets = TensorDataset(
            self.data['x'], self.data['y'],
            self.data['x_attr'], self.data['y_attr']
        )

    def run(self):
        """
        학습 라운드를 반복하고 메트릭을 기록하는 공통 학습 루프입니다.
        """
        rounds = self._resolve_rounds()
        
        # 결과 기록을 위한 워크북 생성
        workbook = xlsxwriter.Workbook('{}_{}.xlsx'.format(self.config.dataset, self.config.pred_steps))
        sheet_round = workbook.add_worksheet('round')
        sheet_round.write(0, 0, 'round_num')
        sheet_round.write(0, 1, '{}_rmse'.format(self.config.agg_model))
        sheet_round.write(0, 2, '{}_mae'.format(self.config.agg_model))

        for rround in range(1, rounds + 1):
            print('**** Round {}/{} ****'.format(rround, rounds))
            train_log = self.train_round(rround)
            train_loss = train_log['log']['rmse'].item()
            train_mae = train_log['log']['mae'].item()

            sheet_round.write(rround, 0, rround)
            sheet_round.write(rround, 1, train_loss)
            sheet_round.write(rround, 2, train_mae)
            print('prediction rmse is: {} '.format(train_loss))

        workbook.close()

    def _resolve_rounds(self):
        """
        Resolve the number of online rounds to execute.

        `rounds <= 0` means use the full stream. For a window size of
        `train_per_num_samples`, the last valid current-evaluation slice starts
        at `max_epoch - train_per_num_samples`.
        """
        max_available_rounds = self.max_epoch - self.train_per_num_samples + 1
        if max_available_rounds <= 0:
            raise ValueError(
                "Dataset is shorter than train_per_num_samples; "
                "no valid online rounds are available."
            )

        rounds = int(getattr(self.config, 'rounds', -1))
        if rounds <= 0:
            return max_available_rounds
        if rounds > max_available_rounds:
            raise ValueError(
                f"rounds={rounds} exceeds the available online rounds "
                f"({max_available_rounds})."
            )
        return rounds

    def train_round(self, rround):
        """
        단일 라운드 학습 플로우를 오케스트레이션합니다.
        """
        # 1. 클라이언트 신규 데이터 갱신 (delayed-label scheduling only)
        self.update_train_data(rround, self.clients)
        
        # 2. 학습 대상 클라이언트 선택 (기본: delayed label이 도착한 전체 eligible client)
        agg_id_list = self.select_clients()
        print('selected clients：', agg_id_list)

        # 3. 로컬 실행 (로컬 학습 및 평가)
        local_logs, agg_state_dict = self.local_execute(agg_id_list)

        # 4. 가중치 병합 (자식 클래스에서 구현해야 하는 부분)
        self.aggregate(agg_state_dict, agg_id_list, rround)

        # 5. 모든 노드의 평가 로그 집계
        agg_log = self.aggregate_local_logs(local_logs)
        
        return {
            'loss': torch.tensor(0).float(),
            'progress_bar': agg_log,
            'log': agg_log
        }

    def update_train_data(self, rround, sample_clients):
        """
        클라이언트의 학습 윈도우를 한 스텝씩 이동시킵니다.
        지연 매칭 프로토콜에 따라 현재 평가용 데이터셋과 지연 학습용 데이터셋을 생성합니다.
        """
        delay = self._resolve_label_delay()
        
        for client in sample_clients:
            client_i = client.client_id
            
            # 1. 현재 평가용 데이터셋 (t 시점)
            client.eval_dataset_current = TensorDataset(
                self.data['x'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['y'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['x_attr'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['y_attr'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :]
            )
            
            # 2. 지연 학습용 데이터셋 (t - delay 시점)
            if rround > delay:
                train_idx = rround - 1 - delay
                client.train_dataset_delayed = TensorDataset(
                    self.data['x'][train_idx:self.train_per_num_samples + train_idx, :, client_i:client_i + 1, :],
                    self.data['y'][train_idx:self.train_per_num_samples + train_idx, :, client_i:client_i + 1, :],
                    self.data['x_attr'][train_idx:self.train_per_num_samples + train_idx, :, client_i:client_i + 1, :],
                    self.data['y_attr'][train_idx:self.train_per_num_samples + train_idx, :, client_i:client_i + 1, :]
                )
            else:
                client.train_dataset_delayed = None

    def _resolve_label_delay(self):
        """
        Resolve delayed-label lag.

        `delay: "pred_steps"` means the full prediction horizon must elapse
        before labels are eligible for adaptation. Numeric strings are accepted
        because YAML defaults drive argparse types.
        """
        delay = getattr(self.config, 'delay', 0)
        if delay == "pred_steps":
            delay = getattr(self.config, 'pred_steps')
        try:
            delay = int(delay)
        except (TypeError, ValueError) as exc:
            raise ValueError('delay must be an integer or "pred_steps".') from exc
        if delay < 0:
            raise ValueError("delay must be non-negative.")
        return delay

    def select_clients(self):
        """
        기본 선택 정책입니다.
        delayed label이 도착해 학습 데이터가 준비된 모든 eligible client를 선택합니다.
        REFOL처럼 drift 기반 partial participation이 필요한 알고리즘은 이 메서드를 오버라이드합니다.
        """
        return [
            client.client_id
            for client in self.clients
            if getattr(client, 'train_dataset_delayed', None) is not None
        ]

    def local_execute(self, agg_id_list):
        """
        모든 클라이언트의 로컬 계산(학습 또는 단순 평가)을 실행하고 로그와 상태를 수집합니다.
        """
        local_logs = []
        agg_state_dict = []
        agg_id_set = set(agg_id_list)
        for idx, client in enumerate(self.clients):
            should_train = client.client_id in agg_id_set
            if should_train:
                # 선택된 노드: 글로벌 모델을 로드하여 학습 진행
                client.local_execute(
                    state_dict_to_load=deepcopy(self.global_model),
                    should_train=True
                )
                agg_state_dict.append(deepcopy(client.local_result['state_dict']))
                local_logs.append(client.local_result['log'])
            else:
                # 선택되지 않은 노드: 로컬 상태를 유지한 채 새 데이터에 대해 평가만 진행
                client.local_execute(state_dict_to_load=None, should_train=False)
                local_logs.append(client.local_result['log'])
        return local_logs, agg_state_dict

    def aggregate(self, local_states, agg_id_list, round):
        """
        [가상 메서드] 서버 사이드 가중치 병합 로직입니다. 자식 클래스에서 오버라이드해야 합니다.
        """
        raise NotImplementedError("Each federated learning method must implement the aggregate() method.")

    def aggregate_local_logs(self, local_logs):
        """
        클라이언트들의 개별 평가 메트릭을 샘플 수 기준으로 가중 평균냅니다.
        """
        agg_log = deepcopy(local_logs[0])
        for k in agg_log:
            agg_log[k] = 0
            for local_log in local_logs:
                if k == 'num_samples':
                    agg_log[k] += local_log[k]
                else:
                    agg_log[k] += local_log[k] * local_log['num_samples']
        for k in agg_log:
            if k != 'num_samples':
                agg_log[k] /= agg_log['num_samples']
        return agg_log
