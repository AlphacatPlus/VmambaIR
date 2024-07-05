
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=7310 \
    --use_env \
    VmambaIR/train.py \
    -opt options/mambaSR11_x4.yml \
    --launcher pytorch 
