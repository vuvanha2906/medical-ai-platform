import torch
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd
)
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
import nibabel as nib
import numpy as np


class MRITumorPredictor:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Đang khởi tạo SwinUNETR cho MRI trên", self.device)

        # 1. Khởi tạo cấu trúc mô hình
        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=24,
            use_checkpoint=True,
        ).to(self.device)

        # 2. CHỈ CẦN NÉM TRỌNG SỐ VÀO ĐÂY:
        # self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # 3. Tiền xử lý (Giống lúc train)
        self.transform = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

    def predict_and_save_mask(self, flair_path, t1_path, t1ce_path, t2_path, output_mask_path):
        # ... Hàm này sẽ ghép 4 file lại, chạy qua mô hình,
        # và lưu kết quả thành 1 file segmentation.nii.gz (mask) để NiiVue hiển thị.
        pass