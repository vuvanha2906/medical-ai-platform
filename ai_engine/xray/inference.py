import torch
import numpy as np
import pydicom
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .model import NIH_DenseNet121
from .grad_cam import XRayGradCAM
import cv2

# Danh sách 14 bệnh chuẩn của NIH theo thứ tự
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

OPTIMAL_THRESHOLDS = {
    'Atelectasis': 0.30,
    'Cardiomegaly': 0.55,
    'Effusion': 0.35,
    'Infiltration': 0.48,
    'Mass': 0.30,
    'Nodule': 0.25,
    'Pneumonia': 0.35,
    'Pneumothorax': 0.30,
    'Consolidation': 0.35,
    'Edema': 0.40,
    'Emphysema': 0.25,
    'Fibrosis': 0.25,
    'Pleural_Thickening': 0.25,
    'Hernia': 0.15
}

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
        original_img = self.load_image(image_path)
        original_img_resized = original_img.resize((224, 224))
        np_img = np.array(original_img_resized)

        input_tensor = self.transform(original_img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            raw_probs = torch.sigmoid(outputs)[0].cpu().numpy()

        # ====================================================
        # 👉 LOGIC HIỆU CHUẨN (CALIBRATION) LÂM SÀNG
        # ====================================================
        calibrated_results = {}
        detected_diseases = {}

        for i, disease in enumerate(DISEASES):
            raw_p = float(raw_probs[i])
            thresh = OPTIMAL_THRESHOLDS[disease]

            # Hiệu chuẩn toán học:
            # Ép giá trị Raw Probability chạy qua điểm neo 50% tại Threshold
            if raw_p >= thresh:
                # Nửa trên: Mapping từ Threshold -> 1.0 thành 0.5 -> 1.0
                calibrated_p = 0.5 + 0.5 * ((raw_p - thresh) / (1.0 - thresh))
                detected_diseases[disease] = calibrated_p
            else:
                # Nửa dưới: Mapping từ 0.0 -> Threshold thành 0.0 -> 0.5
                calibrated_p = 0.5 * (raw_p / thresh)

            calibrated_results[disease] = calibrated_p

        heatmap_paths = {}
        img_normalized = np_img.astype(np.float32) / 255.0

        if not detected_diseases:
            best_class_name = "No Findings"
            best_prob = 0.0
            has_disease = False
            detected_str = "No Findings"

            default_path = f"{output_heatmap_base_path}_original.png"
            Image.fromarray(np_img).save(default_path)
            heatmap_paths["Original"] = default_path

        else:
            has_disease = True

            # Khởi tạo Heatmap cho TẤT CẢ các bệnh vượt ngưỡng
            for disease, prob in detected_diseases.items():
                idx = DISEASES.index(disease)
                targets = [ClassifierOutputTarget(idx)]

                grayscale_cam = self.grad_cam.cam(input_tensor=input_tensor, targets=targets)[0, :]

                # Tight Masking (Lọc rác ngoài phổi và rác yếu)
                body_mask = (np_img.mean(axis=2) > 15).astype(np.float32)
                grayscale_cam = grayscale_cam * body_mask
                grayscale_cam[grayscale_cam < 0.3] = 0

                if grayscale_cam.max() > 0:
                    grayscale_cam = grayscale_cam / grayscale_cam.max()

                heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                alpha = grayscale_cam.copy()
                alpha[alpha > 0] = 0.55
                alpha = np.expand_dims(alpha, axis=-1)

                cam_image = (1 - alpha) * (img_normalized * 255) + alpha * heatmap
                cam_image = np.clip(cam_image, 0, 255).astype(np.uint8)

                specific_path = f"{output_heatmap_base_path}_{disease}.png"
                Image.fromarray(cam_image).save(specific_path)
                heatmap_paths[disease] = specific_path

            best_class_name = max(detected_diseases, key=detected_diseases.get)
            best_prob = detected_diseases[best_class_name]
            # Hiển thị tất cả các bệnh mắc phải ra chuỗi text
            detected_str = ", ".join(list(detected_diseases.keys()))

        return {
            'predicted_class': best_class_name,
            'detected_diseases': detected_str,  # Trả về danh sách bệnh
            'probability': f"{best_prob * 100:.2f}%",
            'all_probabilities': calibrated_results,
            'has_disease': has_disease,
            'heatmaps': heatmap_paths
        }