import numpy as np
import torch
from copy import deepcopy
from base_server import BaseFLServer

class FedAvg(BaseFLServer):

    def __init__(self, config):
        # 부모 클래스 초기화 (데이터 로딩, 클라이언트 생성 준비 등)
        super().__init__(config)

    def boot(self):
        # 1. 공통 준비 실행
        super().boot()
        
        # 2. 추가적인 초기화가 필요하다면 여기에 작성 (예: 추가 신경망 선언)
        # self.agg_layer = nn.Linear(...)
        pass

    def aggregate(self, local_states, agg_id_list, round):

        if not local_states:
            # 이번 라운드에 선택된 클라이언트가 아무도 없다면 기존 글로벌 모델을 유지
            return

        # 1. 각 클라이언트별 학습 데이터 샘플 수(가중치) 계산
        total_samples = 0
        client_samples = []
        for client_id in agg_id_list:
            client = self.clients[client_id]
            # 지연 학습에 사용된 실제 데이터 샘플 수 
            num_samples = len(client.train_dataset_delayed) if client.train_dataset_delayed is not None else 0
            client_samples.append(num_samples)
            total_samples += num_samples

        if total_samples == 0:
            return

        # 2. 템플릿용 뼈대 딕셔너리 복사 (첫 번째 클라이언트의 가중치를 시작점으로 활용)
        agg_state_dict = deepcopy(local_states[0])
        
        # 3. 첫 번째 클라이언트의 가중치에 먼저 자신의 weight를 곱해둠
        weight_0 = client_samples[0] / total_samples
        for key in agg_state_dict.keys():
            # 부동소수점 연산을 위해 잠시 float으로 계산
            agg_state_dict[key] = (agg_state_dict[key].float() * weight_0)

        # 4. 나머지 클라이언트들에 대해 가중 평균 누적
        for idx in range(1, len(local_states)):
            weight = client_samples[idx] / total_samples
            state_dict = local_states[idx]
            
            for key in agg_state_dict.keys():
                # device를 강제하지 않고 타겟 텐서(agg_state_dict[key])의 device를 따라감
                update_val = state_dict[key].to(agg_state_dict[key].device).float() * weight
                agg_state_dict[key] += update_val

        # 5. 원래의 데이터 타입(dtype)으로 복원 (예: 정수형 num_batches_tracked 보존)
        for key in agg_state_dict.keys():
            agg_state_dict[key] = agg_state_dict[key].to(local_states[0][key].dtype)

        # 6. 새로 병합된 가중치로 글로벌 모델 업데이트
        self.global_model = agg_state_dict
