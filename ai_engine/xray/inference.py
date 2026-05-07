import torch
import numpy as np
import pydicom
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from .model import NIH_DenseNet121
from .grad_cam import XRayGradCAM

# Danh sách 14 bệnh chuẩn của NIH theo thứ tự
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

THRESHOLD = 0.3

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

    def load_image(self, image_path: str) -> Image.Image:
        """Đọc file ảnh, hỗ trợ cả DICOM và PNG/JPG/JPEG."""
        if image_path.lower().endswith(('.dcm', '.dicom')):
            # Xử lý DICOM
            dicom_data = pydicom.dcmread(image_path)
            img_array = dicom_data.pixel_array.astype(np.float32)

            # Chuẩn hóa mức xám về [0, 255]
            img_array = img_array - np.min(img_array)
            if np.max(img_array) > 0:
                img_array = img_array / np.max(img_array)
            img_array = (img_array * 255).astype(np.uint8)

            # Xử lý trường hợp MONOCHROME1 (Màu bị đảo ngược: Xương màu đen, nền trắng)
            if hasattr(dicom_data,
                       'PhotometricInterpretation') and dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                img_array = 255 - img_array

            return Image.fromarray(img_array).convert('RGB')
        else:
            # Đọc các định dạng ảnh thông thường
            return Image.open(image_path).convert('RGB')

    def predict_and_explain(self, image_path: str, output_heatmap_base_path: str) -> dict:
        original_img = Image.open(image_path).convert('RGB')
        original_img_resized = original_img.resize((224, 224))
        np_img = np.array(original_img_resized)

        input_tensor = self.transform(original_img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.sigmoid(outputs)[0].cpu().numpy()

        results = {DISEASES[i]: float(probs[i]) for i in range(14)}

        # Lọc các bệnh vượt ngưỡng (Threshold ví dụ = 0.4 hoặc 0.5)
        detected_diseases = {disease: prob for disease, prob in results.items() if prob >= THRESHOLD}

        # Dictionary lưu đường dẫn heatmap cho từng bệnh
        # Format: {"Pneumonia": "/media/heatmaps/study_123_Pneumonia.png"}
        heatmap_paths = {}

        img_normalized = np_img.astype(np.float32) / 255.0

        if not detected_diseases:
            # KHÔNG CÓ BỆNH (No Findings)
            best_class_name = "No Findings"
            best_prob = 0.0
            has_disease = False
            detected_str = "No Findings"

            # Lưu ảnh gốc làm default heatmap
            default_path = f"{output_heatmap_base_path}_original.png"
            Image.fromarray(np_img).save(default_path)
            heatmap_paths["Original"] = default_path

        else:
            # CÓ BỆNH
            has_disease = True

            # SINH HEATMAP RIÊNG CHO TỪNG BỆNH
            for disease, prob in detected_diseases.items():
                idx = DISEASES.index(disease)
                targets = [ClassifierOutputTarget(idx)]

                # Tính toán Grad-CAM cho RIÊNG bệnh này
                grayscale_cam = self.grad_cam.cam(input_tensor=input_tensor, targets=targets)[0, :]

                # Trộn màu (Mặc định GradCAM sẽ dùng dải màu Đỏ-Vàng-Xanh, độ đậm tùy vào grayscale_cam)
                cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

                # Lưu file riêng biệt cho bệnh này
                specific_path = f"{output_heatmap_base_path}_{disease}.png"
                Image.fromarray(cam_image).save(specific_path)

                # Lưu lại đường dẫn
                heatmap_paths[disease] = specific_path

            # Primary Finding vẫn là bệnh có % cao nhất (để hiển thị thẻ bự nhất trên UI)
            best_class_name = max(detected_diseases, key=detected_diseases.get)
            best_prob = detected_diseases[best_class_name]
            detected_str = ", ".join(list(detected_diseases.keys()))

        return {
            'predicted_class': best_class_name,
            'detected_diseases': detected_str,
            'probability': f"{best_prob * 100:.2f}%",
            'all_probabilities': results,
            'has_disease': has_disease,
            'heatmaps': heatmap_paths  # <--- TRẢ VỀ TOÀN BỘ DANH SÁCH ẢNH XAI
        }