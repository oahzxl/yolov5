#python -m torch.distributed.launch --nproc_per_node 3 train.py --img 960 --batch 18 --epochs 100 --data data/truck.yaml --weights weights/yolov5l.pt --cfg models/yolov5l.yaml --device 0,2,1 --sync-bn --hyp data/hyp.finetune.yaml
#python -m torch.distributed.launch --nproc_per_node 3 train.py --img 960 --batch 12 --epochs 100 --data data/truck.yaml --weights weights/yolov5x.pt --cfg models/yolov5x.yaml --device 0,2,1 --sync-bn --hyp data/hyp.finetune.yaml
#python train.py --epochs 10 --data data/truck.yaml --weights weights/yolov5x.pt --img 960 --batch 4 --hyp data/hyp.finetune.yaml --cache --evolve --device 0

for i in 0 1 2; do
  nohup python train.py --epochs 10 --data data/truck.yaml --weights weights/yolov5x.pt --img 960 --batch 4 --hyp data/hyp.finetune.yaml --cache --evolve --device $i > evolve_gpu_$i.log &
done