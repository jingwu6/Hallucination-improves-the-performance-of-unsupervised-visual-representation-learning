import os

# cmd = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python3 DDP_moco.py configs/small/cifar10/moco_center_hallucinate.py'
# cmd = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python3 DDP_moco.py configs/small/cifar100/moco_center_hallucinate.py'
# cmd = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python3 DDP_moco.py configs/small/cifar100/moco_rcrop.py'
# cmd = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python3 DDP_moco.py configs/small/tiny200/moco_center_hallucinate.py'
# cmd = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python3 DDP_moco.py configs/small/stl10/moco_center_hallucinate.py'
# cmd = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python3 DDP_moco.py configs/IN200/mocov2_center_hallucinate.py'


# cmd = 'python DDP_linear.py configs/linear/tiny200_res18.py --load ./checkpoints/small/tiny200/moco_simccrop/last.pth'
# cmd = 'python DDP_linear.py configs/linear/stl10_res18.py --load ./checkpoints/small/stl10/moco_simccrop/last.pth'


os.system(cmd)
