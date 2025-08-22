import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import os

# ============================================================================
# ADVANCED INITIALIZATION AND UTILITY FUNCTIONS
# ============================================================================

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """Proper truncated normal initialization (JAX-style)"""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor

def s(x, epsilon=1e-30):
    """Stablemax function for numerical stability"""
    return torch.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )

def log_stablemax(x, dim=-1):
    """Log stablemax for stable cross entropy"""
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))

def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    """Stablemax cross entropy loss"""
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    
    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
    
    return -torch.where(valid_mask, prediction_logprobs, 0)

# ============================================================================
# ADVANCED LAYER COMPONENTS
# ============================================================================

class CastedLinear(nn.Module):
    """Linear layer with proper initialization and dtype casting"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, weight, bias)

class CastedEmbedding(nn.Module):
    """Embedding layer with proper initialization and dtype casting"""
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """RMS normalization with proper dtype handling"""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

class RMSNorm(nn.Module):
    """Advanced RMS normalization"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.eps) * self.weight.to(x.dtype)

class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)"""
    def __init__(self, hidden_size, max_seq_len=512, base=10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to query and key tensors"""
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class MultiHeadAttention(nn.Module):
    """Advanced multi-head attention with RoPE"""
    def __init__(self, hidden_size, num_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x, cos_sin=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention computation (simplified flash attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context)

class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        intermediate_size = int(expansion * hidden_size * 2 / 3)
        # Round to nearest multiple of 256 for efficiency
        intermediate_size = ((intermediate_size + 255) // 256) * 256
        
        self.gate_up_proj = CastedLinear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = CastedLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class TransformerBlock(nn.Module):
    """Advanced Transformer block with Post-Norm architecture"""
    def __init__(self, hidden_size, num_heads, expansion=4.0, causal=False):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, causal=causal)
        self.feed_forward = SwiGLU(hidden_size, expansion)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        
    def forward(self, x, cos_sin=None):
        # Post-Norm: apply normalization after residual connection
        attn_out = self.attention(x, cos_sin)
        x = self.norm1(x + attn_out)
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

# ============================================================================
# STATE MANAGEMENT DATACLASSES
# ============================================================================

@dataclass
class HRMInnerCarry:
    """Inner carry state for HRM modules"""
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class HRMCarry:
    """Complete carry state including ACT information"""
    inner_carry: HRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

# ============================================================================
# RECURRENT MODULES
# ============================================================================

class RecurrentModule(nn.Module):
    """Advanced recurrent module with multiple transformer layers"""
    def __init__(self, hidden_size, num_layers, num_heads, expansion=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, expansion)
            for _ in range(num_layers)
        ])
        
    def forward(self, *inputs, cos_sin=None):
        # Combine inputs via element-wise addition
        x = sum(inputs)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, cos_sin=cos_sin)
            
        return x

# ============================================================================
# MAIN HRM MODEL WITH ADVANCED ACT
# ============================================================================

class HierarchicalReasoningModel(nn.Module):
    """
    Advanced Hierarchical Reasoning Model with sophisticated ACT
    """
    def __init__(
        self, 
        vocab_size=100,  # REDUCED: From 1000 to 100 for smaller vocabulary
        hidden_size=256, 
        num_heads=8,
        L_layers=3,
        H_layers=3,
        expansion=2.5,
        max_seq_len=64,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        forward_dtype=torch.float32
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.forward_dtype = forward_dtype
        
        # Input and output networks
        embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / embed_scale
        
        self.input_embedding = CastedEmbedding(vocab_size, hidden_size, embed_init_std, forward_dtype)
        self.embed_scale = embed_scale
        self.output_head = CastedLinear(hidden_size, vocab_size, bias=False)
        
        # Q-heads for ACT
        self.q_head = CastedLinear(hidden_size, 2, bias=True)
        
        # Initialize Q-head for faster learning
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)  # Bias towards continuing initially
        
        # Positional encoding
        self.rope = RotaryPositionalEncoding(num_heads and hidden_size // num_heads or hidden_size, max_seq_len)
        
        # Recurrent modules
        self.L_module = RecurrentModule(hidden_size, L_layers, num_heads, expansion)
        self.H_module = RecurrentModule(hidden_size, H_layers, num_heads, expansion)
        
        # Initial states with float32
        self.register_buffer('z0_L', trunc_normal_init_(torch.empty(1, 1, hidden_size, dtype=forward_dtype), std=1))
        self.register_buffer('z0_H', trunc_normal_init_(torch.empty(1, 1, hidden_size, dtype=forward_dtype), std=1))
        
    def initial_carry(self, batch_size: int, seq_len: int, device: torch.device):
        """Initialize carry state for a batch"""
        return HRMCarry(
            inner_carry=HRMInnerCarry(
                z_H=self.z0_H.expand(batch_size, seq_len, -1).to(device),
                z_L=self.z0_L.expand(batch_size, seq_len, -1).to(device)
            ),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),  # Start halted
            current_data={}
        )
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: HRMCarry) -> HRMCarry:
        """Reset carry state where indicated by reset_flag"""
        batch_size, seq_len = carry.inner_carry.z_H.shape[:2]
        device = carry.inner_carry.z_H.device
        
        new_z_H = torch.where(
            reset_flag.view(-1, 1, 1), 
            self.z0_H.expand(batch_size, seq_len, -1).to(device), 
            carry.inner_carry.z_H
        )
        new_z_L = torch.where(
            reset_flag.view(-1, 1, 1), 
            self.z0_L.expand(batch_size, seq_len, -1).to(device), 
            carry.inner_carry.z_L
        )
        
        return HRMCarry(
            inner_carry=HRMInnerCarry(z_H=new_z_H, z_L=new_z_L),
            steps=carry.steps,
            halted=carry.halted,
            current_data=carry.current_data
        )
    
    def forward_inner(self, carry: HRMCarry, batch: Dict[str, torch.Tensor], N=2, T=2):
        """Inner forward pass with one-step gradient approximation"""
        x = batch['inputs']
        batch_size, seq_len = x.shape
        device = x.device
        
        # Input embedding with scaling
        x_emb = self.input_embedding(x) * self.embed_scale
        
        # Get RoPE embeddings
        cos, sin = self.rope(x_emb, seq_len)
        cos_sin = (cos, sin)
        
        # Get current states and ensure they're on the correct device
        zL = carry.inner_carry.z_L.to(device)
        zH = carry.inner_carry.z_H.to(device)
        
        # Ensure states have the right batch size
        if zL.shape[0] != batch_size:
            zL = zL.expand(batch_size, seq_len, -1).contiguous()
            zH = zH.expand(batch_size, seq_len, -1).contiguous()
        
        # Main computation with gradient stopping (one-step approximation)
        with torch.no_grad():
            for i in range(N * T - 1):
                # Low-level update
                zL = self.L_module(zL, zH, x_emb, cos_sin=cos_sin)
                
                # High-level update every T steps
                if (i + 1) % T == 0:
                    zH = self.H_module(zH, zL, cos_sin=cos_sin)
        
        # Final updates with gradients (1-step approximation)
        zL = self.L_module(zL, zH, x_emb, cos_sin=cos_sin)
        zH = self.H_module(zH, zL, cos_sin=cos_sin)
        
        # Output prediction
        output = self.output_head(zH)
        
        # Q-values for ACT (using mean pooling for sequence-level decision)
        q_logits = self.q_head(zH.mean(dim=1)).to(torch.float32)
        q_halt_logits, q_continue_logits = q_logits[..., 0], q_logits[..., 1]
        
        # Create new carry (detached for next iteration)
        new_carry = HRMCarry(
            inner_carry=HRMInnerCarry(z_H=zH.detach(), z_L=zL.detach()),
            steps=carry.steps.to(device),
            halted=carry.halted.to(device),
            current_data=carry.current_data
        )
        
        return new_carry, output, q_halt_logits, q_continue_logits
    
    def forward(self, x, labels=None, N=2, T=2, use_deep_supervision=True):
        """
        Full forward pass with ACT and deep supervision
        
        Args:
            x: Input tensor [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len] (optional)
            N: Number of high-level cycles
            T: Number of low-level timesteps per cycle
            use_deep_supervision: Whether to use deep supervision training
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Initialize carry
        carry = self.initial_carry(batch_size, seq_len, device)
        
        # Prepare batch
        batch = {'inputs': x}
        if labels is not None:
            batch['labels'] = labels
        
        if self.training and use_deep_supervision:
            return self._forward_with_deep_supervision(carry, batch, N, T)
        else:
            return self._forward_inference(carry, batch, N, T)
    
    def _forward_with_deep_supervision(self, carry: HRMCarry, batch: Dict[str, torch.Tensor], N: int, T: int):
        """Forward pass with deep supervision for training"""
        all_outputs = []
        all_q_halt = []
        all_q_continue = []
        all_losses = []
        
        # Reset carry for new sequence
        carry = self.reset_carry(carry.halted, carry)
        carry.current_data = {k: v.clone() for k, v in batch.items()}
        
        max_segments = self.halt_max_steps
        
        for segment in range(max_segments):
            # Forward pass
            carry, output, q_halt, q_continue = self.forward_inner(carry, carry.current_data, N, T)
            
            all_outputs.append(output)
            all_q_halt.append(q_halt)
            all_q_continue.append(q_continue)
            
            # Compute loss for this segment if labels available
            if 'labels' in batch:
                # LM loss
                lm_loss = stablemax_cross_entropy(
                    output.view(-1, self.vocab_size), 
                    batch['labels'].view(-1)
                ).view(batch['labels'].shape).mean()
                
                # Q-learning targets
                with torch.no_grad():
                    # Compute correctness
                    predictions = torch.argmax(output, dim=-1)
                    is_correct_bool = (predictions == batch['labels']).all(dim=1)  # Keep as boolean
                    is_correct_float = is_correct_bool.float()  # Convert to float for loss
                    
                    # Q-learning targets
                    q_halt_target = is_correct_float
                    
                    # For continue, we need next step Q-values (bootstrapping)
                    if segment < max_segments - 1:
                        # Simplified: assume continuing is better if not correct yet
                        q_continue_target = torch.where(is_correct_bool, q_halt, torch.sigmoid(q_continue))
                    else:
                        q_continue_target = q_halt  # Must halt at end
                
                # Q losses
                q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, q_halt_target)
                q_continue_loss = F.binary_cross_entropy_with_logits(q_continue, q_continue_target)
                
                # Combined loss
                total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
                all_losses.append(total_loss)
            
            # Update step count
            carry.steps = carry.steps + 1
            
            # Determine halting (simplified for training)
            if segment >= max_segments - 1:
                break
            
            # Detach carry for next segment
            carry.inner_carry.z_H = carry.inner_carry.z_H.detach()
            carry.inner_carry.z_L = carry.inner_carry.z_L.detach()
        
        # Return final outputs and losses
        final_output = all_outputs[-1]
        total_loss = sum(all_losses) / len(all_losses) if all_losses else None
        
        return {
            'logits': final_output,
            'loss': total_loss,
            'q_halt_logits': all_q_halt[-1],
            'q_continue_logits': all_q_continue[-1],
            'num_segments': len(all_outputs)
        }
    
    def _forward_inference(self, carry: HRMCarry, batch: Dict[str, torch.Tensor], N: int, T: int):
        """Forward pass for inference with ACT halting"""
        carry = self.reset_carry(carry.halted, carry)
        carry.current_data = {k: v.clone() for k, v in batch.items()}
        
        for segment in range(self.halt_max_steps):
            carry, output, q_halt, q_continue = self.forward_inner(carry, carry.current_data, N, T)
            
            carry.steps = carry.steps + 1
            
            # Halting decision
            should_halt = (q_halt > q_continue) | (carry.steps >= self.halt_max_steps)
            
            if should_halt.all():
                break
            
            # Detach for next iteration
            carry.inner_carry.z_H = carry.inner_carry.z_H.detach()
            carry.inner_carry.z_L = carry.inner_carry.z_L.detach()
        
        return {
            'logits': output,
            'q_halt_logits': q_halt,
            'q_continue_logits': q_continue,
            'steps_taken': carry.steps,
            'num_segments': segment + 1
        }
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# MODEL CREATION AND TRAINING UTILITIES WITH FLEXIBLE VOCAB SIZE
# ============================================================================

def create_hrm_model_advanced(
    vocab_size=100,  # OPTIMIZED: Default to 100 instead of 1000 for smaller vocabulary
    hidden_size=256,
    num_heads=8,
    L_layers=3,
    H_layers=3,
    expansion=2.5,
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    forward_dtype=torch.float32
):
    """Create advanced HRM model with optimized vocabulary size"""
    model = HierarchicalReasoningModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        L_layers=L_layers,
        H_layers=H_layers,
        expansion=expansion,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=halt_exploration_prob,
        forward_dtype=forward_dtype
    )
    
    total_params = model.count_parameters()
    
    # Calculate parameter savings vs 1000 vocab
    embedding_params_current = vocab_size * hidden_size
    output_params_current = hidden_size * vocab_size
    total_vocab_params_current = embedding_params_current + output_params_current
    
    embedding_params_1000 = 1000 * hidden_size  
    output_params_1000 = hidden_size * 1000
    total_vocab_params_1000 = embedding_params_1000 + output_params_1000
    
    savings = total_vocab_params_1000 - total_vocab_params_current
    
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params / 1e6:.2f}M parameters")
    if vocab_size != 1000:
        print(f"💰 Parameter savings vs vocab_size=1000: {savings:,} ({savings/total_vocab_params_1000*100:.1f}%)")
    print(f"Forward dtype: {forward_dtype}")
    
    return model

def train_hrm_advanced(model, train_loader, num_epochs=10, lr=1e-3):
    """
    Advanced training with deep supervision and proper loss handling
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_segments = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass with deep supervision
            outputs = model(x, labels=y, use_deep_supervision=True)
            
            if outputs['loss'] is not None:
                loss = outputs['loss']
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                total_segments += outputs['num_segments']
        
        avg_loss = total_loss / len(train_loader)
        avg_segments = total_segments / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Avg Segments: {avg_segments:.2f}")

# ============================================================================
# EXAMPLE USAGE AND MODEL SAVING
# ============================================================================

if __name__ == "__main__":
    print("🔧 GPU Setup for GTX 960M:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    
    default_dtype = torch.float32
    print(f"Using dtype: {default_dtype} (safe for all GPUs)")
    
    # Create model with optimized vocabulary size for your use case
    print("\n=== Creating Optimized HRM Model ===")
    model = create_hrm_model_advanced( 
        vocab_size=100,  # OPTIMIZED: Set to 100 for your tokenizer
        hidden_size=256,
        num_heads=8,
        L_layers=3,  
        H_layers=3,
        expansion=2.5, 
        halt_max_steps=8,
        halt_exploration_prob=0.1,
        forward_dtype=default_dtype
    )
    
    # For comparison, show what the larger model would have been
    print("\n=== Comparison: What 1000 vocab model would have been ===")
    large_model = create_hrm_model_advanced(
        vocab_size=100,
        hidden_size=256,
        num_heads=8,
        L_layers=3,
        H_layers=3,
        expansion=2.5,
        halt_max_steps=8,
        halt_exploration_prob=0.1,
        forward_dtype=default_dtype
    )
    
    # Test the optimized model
    print("\n=== Model Test ===")
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, 100, (batch_size, seq_len))  # Use vocab_size=100
    y = torch.randint(0, 100, (batch_size, seq_len))
    
    try:
        model.train()
        outputs = model(x, labels=y, use_deep_supervision=True)
        print(f"✅ Optimized model working - Loss: {outputs['loss']:.4f}")
        
        # Save the optimized model
        model_data = {
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': 100,  # OPTIMIZED
                'hidden_size': 256,
                'num_heads': 8,
                'L_layers': 3,
                'H_layers': 3,
                'expansion': 2.5,
                'halt_max_steps': 8,
                'halt_exploration_prob': 0.1,
                'forward_dtype': default_dtype
            },
            'total_parameters': model.count_parameters(),
            'model_type': 'HierarchicalReasoningModel_Advanced'  # Keep compatible name
        }
        torch.save(model_data, 'hrm_model_advanced_with_config.pth')  # Keep same filename
        print("✅ Optimized model saved to: hrm_model_advanced_with_config.pth")
        print("🎯 Perfect for your continual_tokenizer.json with <100 vocabulary!")
        print("Ready for training with train_hrm_repo_style.py!")
        
    except Exception as e:
        print(f"❌ Error during model test: {e}")
        import traceback
        traceback.print_exc()