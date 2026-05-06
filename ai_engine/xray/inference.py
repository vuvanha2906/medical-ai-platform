import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .model import NIH_DenseNet121
from .grad_cam import XRayGradCAM

# Danh sách 14 bệnh chuẩn của NIH theo thứ tự
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


class XrayPredictor:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing XrayPredictor on {self.device}...")

        # 1. Khởi tạo model và nạp trọng số bạn vừa train
        self.model = NIH_DenseNet121(num_classes=14)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        # 2. Tiền xử lý ảnh
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. Cài đặt Grad-CAM (Chỉ vào block conv cuối cùng của DenseNet121)
        target_layer = self.model.model.features[-1]
        self.grad_cam = XRayGradCAM(model=self.model, target_layer=target_layer)

    def predict_and_explain(self, image_path: str, output_heatmap_path: str) -> dict:
        """
        Dự đoán bệnh và sinh ra ảnh xAI lưu thẳng vào ổ cứng.
        """
        # Đọc ảnh gốc bằng PIL và resize về 224x224 để vẽ heatmap lên
        original_img = Image.open(image_path).convert('RGB')
        original_img_resized = original_img.resize((224, 224))
        np_img = np.array(original_img_resized)

        # Tiền xử lý thành Tensor
        input_tensor = self.transform(original_img_resized).unsqueeze(0).to(self.device)

        # Chạy mô hình
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # SỬ DỤNG SIGMOID THAY CHO SOFTMAX!
            probs = torch.sigmoid(outputs)[0].cpu().numpy()

        # Tạo từ điển kết quả (Bệnh -> Xác suất)
        results = {DISEASES[i]: float(probs[i]) for i in range(14)}

        # Tìm bệnh có xác suất cao nhất (Primary Finding)
        best_class_idx = np.argmax(probs)
        best_class_name = DISEASES[best_class_idx]
        best_prob = float(probs[best_class_idx])

        # SINH XAI HEATMAP CHO BỆNH CAO NHẤT ĐÓ
        # ClassifierOutputTarget giúp Grad-CAM biết cần tính đạo hàm cho class nào
        targets = [ClassifierOutputTarget(best_class_idx)]
        heatmap_img = self.grad_cam.generate(input_tensor, np_img, targets)

        # Lưu ảnh heatmap ra thư mục media của Django
        Image.fromarray(heatmap_img).save(output_heatmap_path)

        return {
            'predicted_class': best_class_name,
            'probability': f"{best_prob * 100:.2f}%",
            'all_probabilities': results
        }