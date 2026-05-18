import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


class XRayGradCAM:
    def __init__(self, model, target_layer):
        # Khởi tạo Grad-CAM gắn vào layer cuối cùng của model
        self.cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    def generate(self, input_tensor, original_rgb_img, target_category_obj=None):
        """
        Sinh ra ảnh heatmap đè lên ảnh gốc.
        original_rgb_img: Ảnh gốc dạng numpy array (0-255) kích thước 224x224.
        """
        # Tính toán Grad-CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=target_category_obj)[0, :]

        # Chuẩn hóa ảnh gốc về khoảng [0, 1] để show_cam_on_image có thể trộn màu
        img_normalized = original_rgb_img.astype(np.float32) / 255.0

        # Trộn Heatmap vào ảnh gốc
        cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        return cam_image