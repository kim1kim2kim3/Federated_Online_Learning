import numpy as np
import torch
from copy import deepcopy
from base_server import BaseFLServer

class TemplateMethod(BaseFLServer):
    """
    BaseFLServer를 상속받아 새로운 연합학습 병합 알고리즘을 신속하게 구현하기 위한 빈 템플릿입니다.
    """
    def __init__(self, config):
        # 부모 클래스 초기화 (데이터 로딩, 클라이언트 생성 준비 등)
        super().__init__(config)

    def boot(self):
        # 1. 공통 준비 실행
        super().boot()
        
        # 2. 추가적인 초기화가 필요하다면 여기에 작성 (예: 추가 신경망 선언, 어텐션 레이어 등)
        # self.custom_layer = ...
        pass

    def aggregate(self, local_states, agg_id_list, round):
        """
        [필수 구현] 서버가 각 클라이언트 모델의 state_dict들을 병합하여 새로운 글로벌 모델을 구축하는 로직입니다.
        여기에 나만의 병합(Aggregation) 알고리즘을 구현하세요.
        
        Args:
            local_states (list): 선택된 클라이언트들의 로컬 state_dict 리스트.
            agg_id_list (list): 이번 라운드에 선택된 클라이언트 ID 리스트.
            round (int): 현재 학습 라운드 번호.
        """
        if not local_states:
            # 이번 라운드에 선택된 클라이언트가 아무도 없다면 기존 글로벌 모델을 유지
            return

        # ---------------------------------------------------------
        # [여기에 사용자 정의 병합 알고리즘을 구현하세요]
        #
        # 구현 예시:
        # 1. 새로운 빈 가중치 딕셔너리 생성
        # agg_state_dict = deepcopy(local_states[0])
        #
        # 2. 파라미터 연산 (예: 평균, 어텐션 가중치 적용 등)
        # for key in agg_state_dict.keys():
        #     agg_state_dict[key] = ...
        #
        # 3. 계산된 가중치를 글로벌 모델로 업데이트
        # self.global_model = agg_state_dict
        # ---------------------------------------------------------
        
        pass