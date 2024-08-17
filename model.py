# # model.py

import torch
import torch.nn as nn
import config

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int = 3, patch_size:int = 16, embedding_dim:int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
    
    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim:int=768, num_heads:int = 12, attn_dropout:float = 0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
    
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)
    
    def forward(self, x):
        attn_output = self.msa_block(x) + x
        output = self.mlp_block(attn_output)
        return output + attn_output




class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        # Ensure embedding_dim and mlp_size are valid
        assert isinstance(embedding_dim, int) and embedding_dim > 0, "embedding_dim should be a positive integer"
        assert isinstance(mlp_size, int) and mlp_size > 0, "mlp_size should be a positive integer"
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        return self.mlp(x)

class ViT(nn.Module):

    def __init__(self, img_size=224, in_channels=3, patch_size=16, num_transformer_layers=12, 
                 embedding_dim=768, mlp_size=3072, num_heads=12, attn_dropout=0.1, 
                 mlp_dropout=0.1, embedding_dropout=0.1, num_classes=101):
        super().__init__()
        
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size"
        
        self.num_patches = (img_size * img_size) // patch_size**2
        
        # Ensure the embeddings are valid tensors
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # Initialize PatchEmbedding and TransformerEncoderBlock
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoderBlock(
                embedding_dim=embedding_dim, num_heads=num_heads, 
                mlp_size=mlp_size, mlp_dropout=mlp_dropout, attn_dropout=attn_dropout
            ) 
            for _ in range(num_transformer_layers)
        ])
        
        # Initialize the classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]

        # Class token and patch embedding
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        
        # Concatenate class token and add position embedding
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        
        # Apply dropout and pass through the transformer encoder
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        
        # Classify using the classifier
        x = self.classifier(x[:, 0])
        
        return x

