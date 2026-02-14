import torch
import torch.nn as nn
from torchvision import models

class HybridCNNTransformer(nn.Module):
    
    def __init__(
        self,
        num_classes=4,
        feature_dim=512,
        num_heads=8,
        num_layers=3,
        patch_size=1
    ):
        super(HybridCNNTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        
        MLP_RATIO = 4.0
        DROPOUT = 0.1
        
        self.backbone = models.efficientnet_b3(pretrained=False)
        
        self.backbone.avgpool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        
        self.feature_channels = 1536  
        self.feature_map_size = 7     
        self.num_patches = (self.feature_map_size // patch_size) ** 2  
        
        print(f"Backbone: EfficientNet-B3")
        print(f"   Output channels: {self.feature_channels}")
        print(f"   Feature map size: {self.feature_map_size}x{self.feature_map_size}")
        print(f"   Num patches: {self.num_patches}")
        

        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                self.feature_channels,
                feature_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2),  
        )
        

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, feature_dim)  
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        

        self.pre_norm = nn.LayerNorm(feature_dim)
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=int(feature_dim * MLP_RATIO),
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        print(f"Transformer: {num_layers} layers, {num_heads} heads")
        print(f"   Feature dim: {feature_dim}")
        print(f"   MLP ratio: {MLP_RATIO}")
        

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        print(f"Model initialized: {num_classes} classes")
    
    def forward(self, x):

        features = self.backbone.features(x)
        
        patches = self.patch_embed(features)
        
        patches = patches.transpose(1, 2)
        
        batch_size = patches.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        
        patches = patches + self.pos_embed
        
        patches = self.pre_norm(patches)
        
        features = self.transformer_encoder(patches)
        
        cls_output = features[:, 0]
        
        output = self.classifier(cls_output)
        
        return output
    
    def get_feature_maps(self, x):
        return self.backbone.features(x)


def create_model(num_classes=4, device='cpu'):
    model = HybridCNNTransformer(
        num_classes=num_classes,
        feature_dim=512,
        num_heads=8,
        num_layers=3,
        patch_size=1
    )
    
    return model.to(device)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=4, device=device)
    
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    
    print(f"\nTest forward pass:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")