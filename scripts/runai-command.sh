runai submit iris-train   \
  --image registry.rcp.epfl.ch/iris-qwen/iris-qwen:dev   \
  --gpu 1   \
  --pvc aiteam-scratch:/scratch   \
  -e WANDB_API_KEY=93ea3006bbb8c800d529c4bad9f5b2d27a071e43   \
  --command -- bash -c "
    export PYTHONPATH=/scratch/iris/iris_repo/src:\$PYTHONPATH && 
    cd /scratch/iris/iris_repo && 
    python -m iris.cli.finetune.train \
      --config /mnt/aiteam/scratch/iris/iris_repo/configs/vlm/train_mini.yaml \
      --wandb-run-name test-mini-hotfix"
