import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
from typing import Dict
import os
import sys
import cv2
import numpy as np
import traceback 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ml_models.hybrid_model import HybridCNNTransformer 

class MLService:
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.model = self._load_model()
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
        
        print("ML Service initialized successfully")
    
    def _load_model(self):
        
        weights_path = 'models/best_weights_f1.pth'
        
        if not os.path.exists(weights_path):
            print(f"Weights not found at {weights_path}. Initializing model without loading weights.")
            model = HybridCNNTransformer(
                num_classes=4,
                feature_dim=512,
                num_heads=8,
                num_layers=3,      
                patch_size=1
            )
            model = model.to(self.device)
            return model
        
        try:
            print(f"Loading weights from {weights_path}...")
            
            model = HybridCNNTransformer(
                num_classes=4,
                feature_dim=512,
                num_heads=8,
                num_layers=3,      
                patch_size=1
            )
            
            state_dict = torch.load(
                weights_path,
                map_location=self.device,
                weights_only=False
            )
            
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            print("Weights loaded successfully")
            
            return model
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
    
    def _register_hooks(self):

        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'features'):
            target_layer = self.model.backbone.features[-1] 
            
            if target_layer:
                target_layer.register_forward_hook(forward_hook)
                target_layer.register_full_backward_hook(backward_hook) 
                print(f"Grad-CAM hooks registered on: {target_layer.__class__.__name__} (EfficientNet-B3 features[-1])")
            else:
                print("Target layer not found for Grad-CAM on EfficientNet-B3.")
        else:
            print("Model structure missing 'backbone' or 'features' for Grad-CAM registration.")
    
    def predict(self, image_path: str) -> Dict:
        start_time = time.time()
        
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = self.class_names[predicted_idx.item()]
            
            probs_dict = {
                class_name: float(probabilities[i].cpu().numpy())
                for i, class_name in enumerate(self.class_names)
            }
            
            inference_time = int((time.time() - start_time) * 1000)
            
            print(f"Prediction: {predicted_class} ({confidence:.2%}) - {inference_time}ms")
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence.cpu().numpy()),
                'probabilities': probs_dict,
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc() 
            raise
    
    def generate_gradcam(self, image_path: str, output_path: str) -> bool:
        try:
            print(f"Generating Grad-CAM for {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                raise FileNotFoundError(f"Could not read image file at {image_path} using cv2.")
                
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            input_tensor.requires_grad = True
            
            self.model.eval()
            outputs = self.model(input_tensor)
            
            predicted_class = outputs.argmax(dim=1).item()
            
            self.model.zero_grad()
            target = outputs[0, predicted_class]
            target.backward()
            
            if self.gradients is None or self.activations is None:
                print("Grad-CAM hooks didn't capture data. Using simple CAM instead.")
                return self._generate_simple_cam(image_path, output_path)
            
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
            cam = torch.nn.functional.relu(cam)
            
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            cam = cam.cpu().numpy()
            cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            alpha = 0.4
            superimposed = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
            
            result_bgr = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            
            print(f"Grad-CAM saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            traceback.print_exc()
            return False

    def _generate_simple_cam(self, image_path: str, output_path: str) -> bool:

        try:
            print("Generating simple attention map...")
            
            image = Image.open(image_path).convert('RGB')
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                raise FileNotFoundError(f"Could not read image file at {image_path} using cv2.")
                
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            h, w = original_image.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            
            attention = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
            attention = (attention - attention.min()) / (attention.max() - attention.min())
            
            heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            alpha = 0.4
            superimposed = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
            
            result_bgr = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            
            print(f"Simple CAM saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Simple CAM failed: {e}")
            return False

ml_service = MLService()