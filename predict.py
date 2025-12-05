# predict.py
from cog import BasePredictor, Input, Path
import torch
import torch.optim
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import model # Zero-DCE 폴더 안에 있는 model.py를 불러옵니다

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Zero-DCE 모델 초기화
        self.scale_factor = 12
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.DCE_net = model.enhance_net_nopool(self.scale_factor).to(self.device)
        
        # 가중치 파일 로드 (경로 확인 필수)
        weights_path = "snapshots/Epoch99.pth"
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights file not found at {weights_path}")
            
        self.DCE_net.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.DCE_net.eval()

    def predict(
        self,
        image: Path = Input(description="Input image to enhance"),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # 1. 이미지 로드 및 전처리
        data_lowlight = Image.open(str(image)).convert('RGB')
        
        # 이미지를 4의 배수로 리사이징 (모델 구조상 필요할 수 있음, 여기선 원본 크기 유지 시도)
        # 텐서 변환
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1) # HWC -> CHW
        data_lowlight = data_lowlight.cuda().unsqueeze(0) # 배치 차원 추가

        # 2. 추론 (Inference)
        with torch.no_grad():
            _, enhanced_image, _ = self.DCE_net(data_lowlight)

        # 3. 후처리 및 저장
        output_path = Path("/tmp/output.png")
        torchvision.utils.save_image(enhanced_image, output_path)

        return output_path