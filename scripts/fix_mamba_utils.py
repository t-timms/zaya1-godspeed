"""Add cca_state_shape and cca_state_dtype to MambaStateCalculators."""
path = "/home/ttimm/vllm-src/vllm/model_executor/layers/mamba/mamba_utils.py"
with open(path) as f:
    content = f.read()

# Add cca_state_dtype before the last method of MambaStateDtypeCalculator
# Find the class and add before its last method
old_dtype = '''    def kda_state_dtype('''
new_dtype = '''    @staticmethod
    def cca_state_dtype(
        model_dtype: torch.dtype,
        cache_dtype: str,
    ) -> tuple[torch.dtype, ...]:
        """CCA state dtypes: (temporal_state_dtype, conv_state_dtype)."""
        temporal_dtype = model_dtype
        if cache_dtype.startswith("fp8"):
            conv_dtype = torch.float8_e4m3fn
        elif cache_dtype.startswith("fp"):
            conv_dtype = model_dtype
        else:
            conv_dtype = model_dtype
        return (temporal_dtype, conv_dtype)

    def kda_state_dtype('''

if old_dtype in content:
    content = content.replace(old_dtype, new_dtype)
    print("Added cca_state_dtype")
else:
    print("cca_state_dtype pattern not found")

# Add cca_state_shape before kda_state_shape in MambaStateShapeCalculator
old_shape = '''    def kda_state_shape('''
new_shape = '''    @staticmethod
    def cca_state_shape(
        tp_world_size: int,
        conv_kernel_size: int,
        num_k_heads: int,
        num_q_heads: int,
        head_dim: int,
        hidden_size: int,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """CCA state shapes: (conv_state, temporal_state).

        The conv state stores the QK packed values for CCA convolution.
        CCA processes in_out_ch = num_k_heads * head_dim + num_q_heads * head_dim
        dimensions with a kernel of size conv_kernel_size.
        """
        from vllm.distributed.utils import divide
        
        in_out_ch = num_k_heads * head_dim + num_q_heads * head_dim
        conv_dim = divide(in_out_ch, tp_world_size)
        conv_state_shape = (conv_dim, conv_kernel_size)
        
        # temporal state: stores previous hidden states for CCA
        temporal_state_shape = (divide(num_q_heads, tp_world_size), head_dim, num_k_heads)
        
        return (conv_state_shape, temporal_state_shape)

    def kda_state_shape('''

if old_shape in content:
    content = content.replace(old_shape, new_shape)
    print("Added cca_state_shape")
else:
    print("cca_state_shape pattern not found")

with open(path, "w") as f:
    f.write(content)
print("mamba_utils.py updated.")
