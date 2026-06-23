from thop import profile
import torch
import time
import argparse  # 新增：解析命令行参数
import warnings
from net.CIDNet import CIDNet
from net.depth_prior_net import FSNet
from net.depth_boosting import low_light_transformer2
# from net.depth_cidnet_2 import DepthCIDNet
# from net.depth_cidnet_fusion import DepthCIDNet
# from net.depth_cidnet_fusion_4 import RefineNet
from net.depth_mst_3 import MST_Plus_Plus

# ===================== 1. 解析命令行参数（指定GPU ID） =====================
parser = argparse.ArgumentParser(description='Model Latency/Params/FLOPs Test (Specify GPU ID)')
parser.add_argument('--gpu', type=int, default=1, help='GPU device ID to use (default: 0)')
args = parser.parse_args()

# ===================== 2. 配置GPU设备（核心修改） =====================
def setup_device(gpu_id):
    """设置指定的GPU设备，包含有效性校验"""
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available, using CPU instead!")
        return torch.device("cpu")
    
    # 检查指定的GPU ID是否存在
    gpu_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= gpu_count:
        warnings.warn(f"GPU ID {gpu_id} is invalid (only {gpu_count} GPUs available), using GPU 0 instead!")
        gpu_id = 0
    
    # 设置默认GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"✅ Using device: {device} (GPU Name: {torch.cuda.get_device_name(gpu_id)})")
    return device

# 初始化设备
device = setup_device(args.gpu)

# ===================== 3. 模型初始化（绑定指定GPU） =====================
# model = CIDNet().to(device)  
# model = FSNet().to(device)  
# model = RefineNet().to(device)  # 修改：替换为指定device，而非固定'cuda'
model = MST_Plus_Plus().to(device)  # 修改：替换为指定device，而非固定'cuda'
# model = low_light_transformer2().to(device)  

# ===================== 4. 输入张量初始化（绑定指定GPU） =====================
input = torch.rand(1, 3, 256, 256).to(device)  # 修改：使用指定device
depth = torch.rand(1, 1, 256, 256).to(device)  # 修改：使用指定device
# mask = torch.rand(1,1,256,256).to(device)  

# ===================== 5. 耗时计算（保持原有逻辑） =====================
torch.cuda.synchronize()  # 同步CUDA，确保计时准确
model.eval()
time_start = time.time()

with torch.no_grad():  # 新增：推理时禁用梯度计算，加速+节省显存
    _ = model(input, depth)
    # _ = model(input)

torch.cuda.synchronize()
time_end = time.time()
time_sum = time_end - time_start
print(f"\n⏱️ Inference Time: {time_sum:.6f} seconds")

# ===================== 6. 参数量计算（保持原有逻辑） =====================
n_param = sum([p.nelement() for p in model.parameters()])  
n_paras_m = n_param / (2**20)  # 转换为MB
print(f"📊 Model Parameters: {n_paras_m:.4f} M")

# ===================== 7. FLOPs计算（保持原有逻辑） =====================
# 注意：profile需要确保输入张量在指定GPU上
macs, params = profile(model, inputs=(input, depth)) 
# macs, params = profile(model, inputs=(input,)) 
flops_g = macs / (2**30)  # 转换为GFLOPs
print(f"🔥 Model FLOPs: {flops_g:.4f} G")

# 额外：打印GPU显存使用情况（可选）
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(device) / (2**20)
    cached = torch.cuda.memory_reserved(device) / (2**20)
    print(f"🖥️ GPU Memory - Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")