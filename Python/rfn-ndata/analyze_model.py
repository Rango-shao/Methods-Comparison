import torch
from torchinfo import summary
from thop import profile
from net import NestFuse_light2_nodense, Fusion_network

def print_section(title):
    print("\n" + "="*50)
    print(f"🔍 {title}")
    print("="*50)

print_section("1. NestFuse_light2_nodense")
input_shape_nest = (1, 1, 256, 256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor_nest = torch.randn(input_shape_nest).to(device)

model_nest = NestFuse_light2_nodense(
    nb_filter=[64, 112, 160, 208, 256],
    input_nc=1,
    output_nc=1,
    deepsupervision=False
).to(device)

if next(model_nest.parameters()).is_cuda:
    input_tensor_nest = input_tensor_nest.to(device)

summary(model_nest, input_size=input_shape_nest, verbose=2)

flops_nest, params_nest = profile(model_nest, inputs=(input_tensor_nest,), custom_ops={})
print(f"\n📊 NestFuse_light2_nodense")
print(f"    Params: {params_nest / 1e6:.2f} M")
print(f"    FLOPs:  {flops_nest / 1e9:.2f} G")



print_section("2. Fusion_network")
# forward of Fusion_network receives two lists：
#   en_ir = [ir_feat1, ir_feat2, ir_feat3, ir_feat4]  → Each is (1, C, H, W)
#   en_vi = [vi_feat1, vi_feat2, vi_feat3, vi_feat4]
# We construct a simulated multi-scale feature map
# Multi-scale feature map for structural simulation
nC = [64, 128, 256, 512]  # Number of channels per scale
H = [64, 32, 16, 8]       # Spatial height of each scale
W = [64, 32, 16, 8]       # The width of space for each scale

input_features_ir = [
    torch.randn(1, nC[0], H[0], W[0]).to(device),  # scale 1
    torch.randn(1, nC[1], H[1], W[1]).to(device),  # scale 2
    torch.randn(1, nC[2], H[2], W[2]).to(device),  # scale 3
    torch.randn(1, nC[3], H[3], W[3]).to(device),  # scale 4
]

input_features_vi = [
    torch.randn(1, nC[0], H[0], W[0]).to(device),
    torch.randn(1, nC[1], H[1], W[1]).to(device),
    torch.randn(1, nC[2], H[2], W[2]).to(device),
    torch.randn(1, nC[3], H[3], W[3]).to(device),
]

model_fusion = Fusion_network(nC=nC, fs_type='res').to(device)

try:
    summary(model_fusion, input_size=[(1, nC[0], H[0], W[0]), (1, nC[0], H[0], W[0])], verbose=0)
except:
    print("[INFO] torchinfo cannot perfectly display the Fusion_network structure (because the input is a list), it is recommended to focus on FLOPs and Params.\n")

flops_fusion, params_fusion = profile(model_fusion, inputs=(input_features_ir, input_features_vi), custom_ops={})
print(f"\n📊 Fusion_network (RFN)")
print(f"    Params: {params_fusion / 1e6:.2f} M")
print(f"    FLOPs:  {flops_fusion / 1e9:.2f} G")

