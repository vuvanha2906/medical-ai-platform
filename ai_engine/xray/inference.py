import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import densenet121
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
from typing import Dict


class XrayPredictor:
    def __init__(self):
        # Load the pre-trained DenseNet121 model
        self.model = densenet121(weights='IMAGENET1K_V1')
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize Grad-CAM
        # The last layer of densenet121.features is the final convolutional block.
        self.grad_cam = GradCAM(model=self.model, target_layers=[self.model.features[-1]])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess the image for inference."""
        with Image.open(image_path).convert('RGB') as img:
            return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, image_path: str) -> dict:
        """Predict the probabilities of the classes for the given image."""
        with torch.no_grad():
            input_tensor = self.preprocess_image(image_path)
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted_idx = torch.max(probabilities, 1)
            predicted_class = predicted_idx.item()
            class_probabilities = {str(i): prob.item() for i, prob in enumerate(probabilities[0])}

        return {
            'predicted_class': predicted_class,
            'probabilities': class_probabilities
        }


def generate_heatmap(self, image_path: str, output_path: str) -> None:
    """Generate and save the Grad-CAM heatmap for the given image."""
    input_tensor = self.preprocess_image(image_path)
    target_category = None  # Use the predicted class by default

    # Generate the Grad-CAM heatmap
    grayscale_cam = self.grad_cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    # Convert the input tensor back to a NumPy array
    input_image = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    input_image = (input_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    input_image = np.clip(input_image, 0, 255).astype(np.uint8)

    # Overlay the heatmap on the input image
    cam_image = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

    # Save the heatmap
    Image.fromarray(cam_image).save(output_path)