import numpy as np
import torch
from copy import deepcopy
from base_server import BaseFLServer
from models.AggregationGCN import AttGCN

class REFOL(BaseFLServer):
    """
    REFOL: Resource-Efficient Federated Online Learning for Traffic Flow Forecasting
    BaseFLServer를 상속받아 중복 로직을 걷어내고, 공간 인접성을 반영하는 GCN 기반의 병합 알고리즘을 구현합니다.
    """
    def __init__(self, config):
        super().__init__(config)

    def boot(self):
        """
        데이터셋 및 클라이언트 초기화를 부모 클래스에 위임하고, GCN 집계 모델을 추가로 로드합니다.
        """
        super().boot()
        self.gcn = AttGCN()


    def select_clients(self):
        """
        REFOL selection policy: evaluate drift for each client, then train only clients
        whose delayed-label data is available and whose drift check selected them.
        """
        agg_id_list = []
        for client in self.clients:
            client.eval_drift()
            if client.selected and getattr(client, 'train_dataset_delayed', None) is not None:
                agg_id_list.append(client.client_id)
        return agg_id_list

    def aggregate(self, local_states, agg_id_list, round):
        """
        공간 교통 네트워크 구조를 고려한 GCN 기반 가중치 병합을 수행합니다.
        """
        if not local_states:
            # 선택된 클라이언트가 없을 경우 기존 글로벌 모델 유지
            return

        edge_index = np.array(deepcopy(self.data['edge_index']))
        sample_id = np.array(agg_id_list)
        
        # 1. 서브 그래프의 연결 관계 필터링 및 인덱스 리매핑
        mask = np.isin(edge_index, sample_id)
        mask1 = np.isin(np.sum(mask, axis=0), 2)
        edge_index = edge_index[:, mask1]
        
        table = np.zeros(sample_id.max() + 1, np.int64)
        table[sample_id] = np.arange(sample_id.size)
        edge_index = torch.from_numpy(table[edge_index])
        
        # 2. 서버 글로벌 노드 가상 매핑 및 양방향 엣지 설정
        tmp = np.full((sample_id.shape), len(sample_id))
        tmp = np.stack((table[sample_id], tmp))
        tmp = np.hstack((tmp, [[len(sample_id)], [len(sample_id)]]))
        edge_index = torch.from_numpy(np.concatenate((edge_index, tmp), axis=1))

        # 3. 기존 서버 글로벌 가중치를 리스트의 마지막에 추가
        tmp_model = self.global_model
        if tmp_model is None:
            tmp_model = deepcopy(local_states[0])
        local_states.append(tmp_model)

        # 4. 모델 가중치 텐서 평탄화 및 노드 피처 행렬(Feature Matrix) 구성
        local_results = []
        for i, local_train_result in enumerate(local_states):
            for name in local_train_result:
                local_results += local_train_result[name].flatten().tolist()
        local_results = torch.Tensor(local_results).view((len(sample_id) + 1, -1))

        # 5. GCN을 통해 가중치 전파 수행
        self.gcn.to(self.device)
        local_results = self.gcn(
            x=local_results.to(self.device),
            edge_index=edge_index.to(self.device)
        )
        
        # 6. GCN 출력 행렬에서 마지막 노드(서버)의 피처를 추출해 글로벌 모델로 재배치
        global_model = local_results[-1]
        agg_state_dict = {}
        len_start = 0
        for name in local_train_result:
            length = len(local_train_result[name].flatten().tolist())
            agg_state_dict[name] = global_model[len_start:len_start + length].reshape_as(local_train_result[name])
            len_start += length
            
        self.global_model = agg_state_dict