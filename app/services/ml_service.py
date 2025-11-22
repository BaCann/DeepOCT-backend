# app/services/ml_service.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
from typing import Dict
import os
import sys

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ml_models.hybrid_model import HybridCNNTransformer

class MLService:
    def __init__(self):
        """Initialize ML service with trained Hybrid CNN-Transformer model"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        # Class names - EXACT order from training
        self.class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
        # Image preprocessing - EXACT same as training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print("ML Service initialized successfully")
    
    def _load_model(self):
        """Load trained model from weights file"""
        
        # ============================================
        # WEIGHTS FILE PATH
        # ============================================
        weights_path = 'models/best_weights_f1.pth'
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found at {weights_path}\n"
                f"Please download best_weights_f1.pth from Kaggle"
            )
        
        try:
            print(f"📦 Loading weights from {weights_path}...")
            print(f"   File size: {os.path.getsize(weights_path) / (1024*1024):.1f} MB")
            
            # ============================================
            # CREATE MODEL ARCHITECTURE
            # ============================================
            print("Creating model architecture...")
            model = HybridCNNTransformer(
                num_classes=4,
                feature_dim=512,
                num_heads=8,
                num_layers=3,      
                patch_size=1
            )
            
            # ============================================
            # LOAD WEIGHTS
            # ============================================
            print("Loading weights...")
            
            # Load state_dict
            state_dict = torch.load(
                weights_path,
                map_location=self.device,
                weights_only=False
            )
            
            # Check if it's wrapped in a dict or direct state_dict
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                # It's a checkpoint
                print("   Format: Checkpoint")
                model.load_state_dict(state_dict['model_state_dict'])
                
                # Print metadata if available
                if 'epoch' in state_dict:
                    print(f"   ├─ Epoch: {state_dict['epoch']}")
                if 'best_f1' in state_dict:
                    print(f"   └─ Best F1: {state_dict['best_f1']:.4f}")
            else:
                # It's direct state_dict
                print("   Format: State dict")
                model.load_state_dict(state_dict)
            
            # ============================================
            # MOVE TO DEVICE
            # ============================================
            model = model.to(self.device)
            
            print("Weights loaded successfully")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            
            return model
            
        except Exception as e:
            import traceback
            print(f"Error loading weights: {e}")
            print("\n" + "="*60)
            print("FULL ERROR TRACEBACK:")
            print("="*60)
            traceback.print_exc()
            print("="*60)
            
            print("\n💡 Troubleshooting:")
            print("1. Verify weights file:")
            print(f"   ls -lh {weights_path}")
            print("\n2. Check weights structure:")
            print(f"   python -c \"import torch; w=torch.load('{weights_path}', map_location='cpu'); print(type(w))\"")
            print("\n3. Verify model architecture matches training")
            raise
    
    def predict(self, image_path: str) -> Dict:
        """
        Run inference on OCT image
        
        Args:
            image_path: Path to image file
            
        Returns:
            {
                'predicted_class': str,
                'confidence': float,
                'probabilities': dict,
                'inference_time': int (ms)
            }
        """
        start_time = time.time()
        
        try:
            # ============================================
            # 1. LOAD & PREPROCESS IMAGE
            # ============================================
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            print(f"Image loaded: {width}x{height}")
            
            # Apply transformations
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            # Shape: [1, 3, 224, 224]
            
            # ============================================
            # 2. RUN INFERENCE
            # ============================================
            with torch.no_grad():
                outputs = self.model(input_tensor)
                # Shape: [1, 4] - logits
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                # Shape: [4] - probabilities sum to 1.0
            
            # ============================================
            # 3. GET PREDICTION
            # ============================================
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = self.class_names[predicted_idx.item()]
            
            # ============================================
            # 4. FORMAT PROBABILITIES
            # ============================================
            probs_dict = {
                class_name: float(probabilities[i].cpu().numpy())
                for i, class_name in enumerate(self.class_names)
            }
            
            # ============================================
            # 5. CALCULATE INFERENCE TIME
            # ============================================
            inference_time = int((time.time() - start_time) * 1000)  # ms
            
            # ============================================
            # 6. LOG RESULT
            # ============================================
            print(f"Prediction: {predicted_class} ({confidence:.2%})")
            print(f"   Probabilities:")
            for cls, prob in probs_dict.items():
                print(f"     {cls}: {prob:.4f}")
            print(f"Inference time: {inference_time}ms")
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence.cpu().numpy()),
                'probabilities': probs_dict,
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
    
    def generate_gradcam(self, image_path: str, output_path: str) -> str:
        """
        Generate Grad-CAM heatmap visualization
        
        Not implemented yet
        """
        print("Grad-CAM not implemented")
        return None

# Singleton instance
ml_service = MLService()