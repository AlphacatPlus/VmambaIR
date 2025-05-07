
#folder_gt=/mnt/bn/shiyuan-arnold/dataset/DiffIR/SISR/Set5/HR
#folder_restored=/mnt/bn/shiyuan-arnold/code/Mamber/DiffIR/DiffIR-SRGAN/results/test_MambaSISR15GAN3/visualization/Set5

folder_gt=/mnt/bn/shiyuan-arnold/dataset/DiffIR/SISR/Set14/HR
folder_restored=/mnt/bn/shiyuan-arnold/code/Mamber/DiffIR/DiffIR-SRGAN/results/test_MambaSISR15GAN3/visualization/Set14

#folder_gt=/mnt/bn/shiyuan-arnold/dataset/DiffIR/SISR/Urban100/HR
#folder_restored=/mnt/bn/shiyuan-arnold/code/Mamber/DiffIR/DiffIR-SRGAN/results/test_MambaSISR15GAN3/visualization/Urban100

#folder_gt=/mnt/bn/shiyuan-arnold/dataset/DiffIR/SISR/Manga109/HR
#folder_restored=/mnt/bn/shiyuan-arnold/code/Mamber/DiffIR/DiffIR-SRGAN/results/test_MambaSISR15GAN3/visualization/Manga109

#folder_gt=/mnt/bn/shiyuan-arnold/dataset/DiffIR/SISR/General100/HR
#folder_restored=/mnt/bn/shiyuan-arnold/code/VmambaIR/SRGAN/results/test_MambaSISR15GAN3/visualization/General100

#folder_gt=/mnt/bn/shiyuan-arnold/dataset/DiffIR/SISR/DIV2K100/HR
#folder_restored=/mnt/bn/shiyuan-arnold/code/VmambaIR/SRGAN/results/test_MambaSISR15GAN3/visualization/DIV2K100


python3  Metrics/LPIPS.py \
    --folder_gt $folder_gt  \
    --folder_restored $folder_restored

python3  Metric/PSNR.py \
    --folder_gt $folder_gt  \
    --folder_restored $folder_restored
