import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Densenet121 import DenseNet3D

from data.atlas.get_scores import process_batch, prepare_regions
from transformers import ViTModel

##############################
# 1. Helper Functions for Patch Extraction
##############################
def pad_for_overlap(tensor, patch_size=16, overlap=0):
    """
    Pads the volume so that overlapping patch extraction covers the entire volume.
    Expects tensor shape: (B, C, D, H, W)
    """
    stride = patch_size - overlap
    B, C, D, H, W = tensor.shape

    def calc_pad(dim):
        if dim < patch_size:
            return patch_size - dim
        # Compute number of patches along the dimension.
        n_patches = ((dim - patch_size) + stride - 1) // stride + 1
        total = (n_patches - 1) * stride + patch_size
        return total - dim

    pad_D = calc_pad(D)
    pad_H = calc_pad(H)
    pad_W = calc_pad(W)
    # F.pad expects padding order: (W_left, W_right, H_left, H_right, D_left, D_right)
    padding = (0, pad_W, 0, pad_H, 0, pad_D)
    return F.pad(tensor, padding)

def extract_overlapping_patches(volume, patch_size=16, overlap=1):
    """
    Extracts overlapping 3D patches from a volume.
    
    Args:
        volume: Tensor of shape (B, C, D, H, W)
        patch_size: Cube patch size (default=16)
        overlap: Number of voxels overlapping (default=2)
        
    Returns:
        patches: Tensor of shape (num_patches, C, patch_size, patch_size, patch_size)
        grid_size: Tuple (nD, nH, nW) indicating number of patches per dimension.
    """
    stride = patch_size - overlap  # e.g. 16 - 2 = 14
    B, C, D, H, W = volume.shape
    patches = volume.unfold(2, patch_size, stride) \
                    .unfold(3, patch_size, stride) \
                    .unfold(4, patch_size, stride)
    # New shape: (B, C, nD, nH, nW, patch_size, patch_size, patch_size)
    nD, nH, nW = patches.size(2), patches.size(3), patches.size(4)
    patches = patches.contiguous().view(B * nD * nH * nW, C, patch_size, patch_size, patch_size)
    return patches, (nD, nH, nW)


##############################
# Custom Multi-Head Attention Implementation
##############################
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # x: (B, N, E)
        B, N, E = x.shape
        # Compute Q, K, V together and reshape
        qkv = self.qkv(x)  # (B, N, 3*E)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, N, head_dim)
        
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, N, N)
        if attn_mask is not None:
            # Expect attn_mask of shape (B * num_heads, N, N)
            # Reshape it to (B, num_heads, N, N)
            attn_mask = attn_mask.reshape(B, self.num_heads, N, N)
            attn_scores = attn_scores + attn_mask
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, E)
        out = self.out_proj(out)
        return out, attn_probs


class TransformerBlock(nn.Module):
    """
    One Transformer Encoder block:
      - LayerNorm
      - Multi-Head Self-Attention (nn.MultiheadAttention)
      - Add & Norm
      - MLP
      - Add & Norm
    """
    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=2.0, dropout=0.1, initial_alpha=0.5, debug=False):
        super().__init__()
        self.debug = debug
        self.num_heads = num_heads
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # Learnable parameters
        self.w2 = nn.Parameter(torch.tensor(1.0))  # Initialize w2
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))  # Initialize alpha as learnable, centered around zero


    def get_bias(self, M):
        alpha_scaled = (torch.tanh(self.alpha + 1e-7) + 1) / 2
        w2 = F.softplus(self.w2 + 1e-7)
        w1 = alpha_scaled * w2

        # Separate the scores of 1 and 2
        mask_1 = (M == 1.).float()
        mask_2 = (M == 2.).float()
         
        # Apply weights to the scores of 1 and 2
        M_weighted = w1 * mask_1 + w2 * mask_2

        attn_bias = torch.bmm(M_weighted, M_weighted.transpose(1, 2))
        normalized_learned_scores = F.normalize(attn_bias, p=2, dim=(1,2))#, dim=-1)
        # Prepare attn_mask for nn.MultiheadAttention
        attn_bias = normalized_learned_scores.repeat_interleave(self.num_heads, dim=0)  # Shape: (batch_size * num_heads, num_query, num_k)
        
        return 50*attn_bias

    def forward(self, x, M):
        """
        x: (B, N, E)
        """
        if self.debug:
            print("[TransformerBlock] Input:", x.shape)
        attn_bias = self.get_bias(M)

        y = self.ln1(x)
        y, _ = self.mhsa(y, y, y, attn_mask=attn_bias)  # Self-attention
        x = x + self.dropout(y)    # Residual

        if self.debug:
            print("[TransformerBlock] After MHSA:", x.shape)

        y = self.ln2(x)
        y = self.mlp(y)
        x = x + self.dropout(y)    # Residual

        if self.debug:
            print("[TransformerBlock] After MLP:", x.shape)

        return x

##############################
# 3. End-to-End Model: SynthSegViT
#    (Takes input of shape (B, C, D, H, W), performs padding, overlapping patch extraction,
#     token embedding, adds learnable positional embeddings, and processes tokens via a Transformer)
##############################
class ViT(nn.Module):
    def __init__(self, fixed_input_shape, num_layers=6, num_heads=8, mlp_dim=1024,
                 dropout=0.1, num_classes=10, embed_dim=512):
        """
        End-to-end model that accepts an input of shape (B, C, D, H, W).
        fixed_input_shape: tuple (C, D, H, W) of expected input (without batch).
        """
        super(ViT, self).__init__()
        self.fixed_input_shape = fixed_input_shape
        self.patch_size = 16
        self.overlap = 0
        self.stride = self.patch_size - self.overlap
        self.embed_dim = embed_dim

        # Patch encoder 
        self.backbone = DenseNet3D(
            init_channels=64,
            growth_rate=32, #32
            block_layers=(6,12,24,16),
            out_dim=embed_dim
        )

        self.num_tokens = 2016
        grid_size = (12, 14, 12)  # For 2016 tokens
        nD, nH, nW = grid_size
        coordinates = []
        for idx in range(self.num_tokens):
            d = idx // (nH * nW)
            rem = idx % (nH * nW)
            h = rem // nW
            w = rem % nW
            coordinates.append([
                d / (nD - 1) if nD > 1 else 0.0,
                h / (nH - 1) if nH > 1 else 0.0,
                w / (nW - 1) if nW > 1 else 0.0
            ])
        # Convert to tensor
        self.register_buffer("coordinates", torch.tensor(coordinates, dtype=torch.float32))  # [2016, 3]
        self.pos_layer = nn.Linear(3, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim//embed_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(self.num_tokens*embed_dim, num_classes)


        self.high_relevance_mask = prepare_regions()
        #for atlas scores
        self.max_pool = nn.MaxPool3d(kernel_size=self.patch_size, stride=self.patch_size)

        self.load_huggingface_weights()
        
    def forward(self, x):
        """
        x: Tensor of shape (B, C, D, H, W)
        """
        B = x.shape[0]

        atlas_data_scores = process_batch(x, self.high_relevance_mask.unsqueeze(0), x.device)
                
        # Pad input
        x = pad_for_overlap(x, patch_size=self.patch_size, overlap=self.overlap)
        atlas_data_scores = pad_for_overlap(atlas_data_scores, patch_size=self.patch_size, overlap=self.overlap)
        # Extract overlapping patches
        patches, _ = extract_overlapping_patches(x, patch_size=self.patch_size, overlap=self.overlap)
        M = self.max_pool(atlas_data_scores.float())
        M = M.reshape(B,-1).unsqueeze(-1) # Shape: [B, total_patches, 1]

        # Get token embeddings from patch encoder
        token_embeddings = self.backbone(patches)
        # Reshape to (B, num_tokens, embed_dim)
        tokens = token_embeddings.view(B, self.num_tokens, self.embed_dim)
        # Add learnable positional embeddings
        pos_embed = self.pos_layer(self.coordinates).to(tokens.device)  # [2016, embed_dim]
        x = tokens + pos_embed.unsqueeze(0)  # Broadcast across batch

        for i, block in enumerate(self.blocks):
            x = block(x, M)
            
        tokens = self.norm(x)

        logits = self.head(tokens.reshape(B,-1))
        return logits

    def load_huggingface_weights(self, hf_model_name="google/vit-base-patch16-224-in21k"):
        """
        Load weights from a Hugging Face ViT model into this custom ViT class.
        This assumes your TransformerBlock layout and dimensions match those
        in HF’s implementation. Mismatched shapes will raise errors and may need
        careful handling or resizing.
        """
        # 1. Load HF ViT
        hf_vit = ViTModel.from_pretrained(hf_model_name)
        hf_encoder = hf_vit.encoder
        hf_embeddings = hf_vit.embeddings
        hf_embedding_dim = hf_vit.config.hidden_size  # e.g., 768 for vit-base
        
        
        # 3. Copy each TransformerBlock’s weights
        if len(self.blocks) != len(hf_encoder.layer):
            raise ValueError(
                f"Number of transformer blocks differ: "
                f"Custom has {len(self.blocks)}, HF has {len(hf_encoder.layer)}."
            )
        
        for i, block in enumerate(self.blocks):
            hf_block = hf_encoder.layer[i]
            
            # ----- LayerNorms -----
            # Hugging Face uses "layernorm_before" and "layernorm_after" in each block
            block.ln1.weight.data.copy_(hf_block.layernorm_before.weight)
            block.ln1.bias.data.copy_(hf_block.layernorm_before.bias)

            block.ln2.weight.data.copy_(hf_block.layernorm_after.weight)
            block.ln2.bias.data.copy_(hf_block.layernorm_after.bias)
            
            q_w = hf_block.attention.attention.query.weight
            k_w = hf_block.attention.attention.key.weight
            v_w = hf_block.attention.attention.value.weight
            in_proj_weight = torch.cat([q_w, k_w, v_w], dim=0)
            
            q_b = hf_block.attention.attention.query.bias
            k_b = hf_block.attention.attention.key.bias
            v_b = hf_block.attention.attention.value.bias
            in_proj_bias = torch.cat([q_b, k_b, v_b], dim=0)
            
            block.mhsa.in_proj_weight.data.copy_(in_proj_weight)
            block.mhsa.in_proj_bias.data.copy_(in_proj_bias)
            
            # The output projection of the attention
            block.mhsa.out_proj.weight.data.copy_(hf_block.attention.output.dense.weight)
            block.mhsa.out_proj.bias.data.copy_(hf_block.attention.output.dense.bias)
            
            # ----- MLP -----
            # HF’s intermediate.dense => first Linear in MLP
            # HF’s output.dense => second Linear in MLP
            block.mlp[0].weight.data.copy_(hf_block.intermediate.dense.weight)
            block.mlp[0].bias.data.copy_(hf_block.intermediate.dense.bias)
            
            block.mlp[2].weight.data.copy_(hf_block.output.dense.weight)
            block.mlp[2].bias.data.copy_(hf_block.output.dense.bias)
        
        # 4. Final layer norm in HF model
        if hasattr(hf_vit, "layernorm"):
            self.norm.weight.data.copy_(hf_vit.layernorm.weight)
            self.norm.bias.data.copy_(hf_vit.layernorm.bias)
        else:
            raise ValueError("No final layernorm found in HF model (vit.layernorm).")
        
        del hf_block, hf_embedding_dim, hf_embeddings, hf_encoder, hf_vit
        print(f"Loaded Hugging Face ViT weights from '{hf_model_name}' successfully.")


##############################
# 4. Main Function
##############################
def main():
    # Fixed input shape: (C, D, H, W) e.g. (1, 182, 218, 182)
    fixed_input_shape = (1, 182, 218, 182)
    # Create the end-to-end model
    model = ViT(fixed_input_shape=fixed_input_shape,
                        num_layers=12, num_heads=12, mlp_dim=768*4,
                        dropout=0.1, num_classes=2, embed_dim=768)
    model.load_huggingface_weights()
    
    # Create a dummy 3D scan with fixed input shape: (B, C, D, H, W)
    dummy_scan = torch.randn(1, *fixed_input_shape)
    # Forward pass through the end-to-end model
    logits = model(dummy_scan)
    print("Model input shape:", dummy_scan.shape)
    print("Model output logits shape:", logits.shape)

if __name__ == '__main__':
    main()
