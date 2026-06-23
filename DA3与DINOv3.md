# DA3 + DINOv3 先验生成

cd /root/Desktop/TGRS
conda activate tgrs
export HF_ENDPOINT=https://hf-mirror.com 

python scripts/prepare_depth.py --dataset isaid --skip-high-depth
python scripts/prepare_depth.py --dataset isaid --root data_dir/iSAID-dark

python scripts/prepare_depth.py --dataset lol --skip-high-depth
python scripts/prepare_depth.py --dataset lol --root data_dir/LOL-v1

cd /root/Desktop/TGRS
conda activate tgrs

python scripts/cache_dinov3_features.py --hub_local ./dinov3-main --weights ./weights_dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth --input_dir ./data_dir/LOL-v1/our485/low --output_dir ./cache_dinov3/our485_low --model dinov3_vitl16 --img_size 224 --device cuda:0

python scripts/cache_dinov3_features.py --hub_local ./dinov3-main --weights ./weights_dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth --input_dir ./data_dir/LOL-v1/eval15/low --output_dir ./cache_dinov3/eval15_low --model dinov3_vitl16 --img_size 224 --device cuda:0



python -u scripts/cache_dinov3_features.py \
  --hub_local /root/Desktop/TGRS/dinov3-main \
  --weights /root/Desktop/TGRS/weights_dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --model dinov3_vitl16 \
  --input_dir /root/Desktop/TGRS/data_dir/LOL-v1/our485/low \
  --output_dir /root/Desktop/TGRS/cache_dinov3/LOLv1_vitl16/train \
  --resize_mode short_side \
  --img_size 224 \
  --device cuda:0


python -u scripts/cache_dinov3_features.py \
  --hub_local /root/Desktop/TGRS/dinov3-main \
  --weights /root/Desktop/TGRS/weights_dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --model dinov3_vitl16 \
  --input_dir /root/Desktop/TGRS/data_dir/LOL-v1/eval15/low \
  --output_dir /root/Desktop/TGRS/cache_dinov3/LOLv1_vitl16/val \
  --resize_mode short_side \
  --img_size 224 \
  --device cuda:0