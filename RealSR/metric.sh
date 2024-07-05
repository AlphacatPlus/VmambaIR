
#folder_gt=/mnt/bn/shiyuan-arnold/dataset/NTIRE2020/track1-valid-gt
#folder_restored=/mnt/bn/shiyuan-arnold/code/VmambaIR/RealSR/results/test_mambaSR11GAN2_archived_20240623_200741/visualization/NTIRE2020-Track1

folder_gt=/mnt/bn/shiyuan-arnold/dataset/AIM19/AIM19/valid-gt-clean
#folder_restored=/mnt/bn/shiyuan-arnold/code/Mamber/DiffIR/DiffIR-RealSR/results/test_mambaSR11GAN_AIM/visualization/AIM19
folder_restored=/mnt/bn/shiyuan-arnold/code/VmambaIR/RealSR/results/test_mambaSR11GAN2/visualization/AIM19

python3  Metric/LPIPS.py \
    --folder_gt $folder_gt  \
    --folder_restored $folder_restored

python3  Metric/PSNR.py \
    --folder_gt $folder_gt  \
    --folder_restored $folder_restored
