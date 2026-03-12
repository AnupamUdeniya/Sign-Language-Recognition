import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class HybridASLModel(nn.Module):
    def __init__(
        self,
        num_classes=29,
        embed_dim=768,
        num_heads=8,
        num_layers=4,
        pretrained_backbone=True
    ):
        super().__init__()

        # -------- CNN Backbone --------
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        try:
            resnet = models.resnet50(weights=weights)
        except Exception as exc:
            print(f"Warning: could not load pretrained ResNet50 weights ({exc}).")
            print("Falling back to randomly initialized ResNet50 backbone.")
            resnet = models.resnet50(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool + fc

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Project CNN channels to transformer embedding dim
        self.patch_proj = nn.Linear(2048, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 49 + 1, embed_dim))

        # -------- Transformer Encoder --------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # CNN vector projection for fusion
        self.cnn_proj = nn.Linear(2048, embed_dim)

        # Fusion + classifier
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        # CNN feature extraction
        features = self.cnn(x)  # (B, 2048, 7, 7)

        B, C, H, W = features.shape

        # -------- Transformer tokens --------
        tokens = features.flatten(2).transpose(1, 2)  # (B, 49, 2048)
        tokens = self.patch_proj(tokens)  # (B, 49, 768)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        tokens = tokens + self.pos_embed

        transformer_out = self.transformer(tokens)

        cls_output = transformer_out[:, 0]  # CLS token

        # -------- CNN global feature --------
        cnn_vec = self.gap(features).flatten(1)
        cnn_vec = self.cnn_proj(cnn_vec)

        # -------- Fusion --------
        fused = torch.cat([cls_output, cnn_vec], dim=1)
        fused = self.fusion(fused)

        out = self.classifier(fused)

        return out


def get_model(num_classes=29, pretrained_backbone=True):
    return HybridASLModel(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone
    )
