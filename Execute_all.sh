echo launching
conda init

echo opening environment
chdir "~/islt_directml/"

conda activate slt_directml
echo environment activated

echo batch 8
python -m signjoey train configs/8head/sign_8head_8batch.yaml && python -m signjoey train configs/8head/sign_8head_16batch.yaml && python -m signjoey train configs/8head/sign_8head_32batch.yaml && python -m signjoey train configs/8head/sign_8head_64batch.yaml && python -m signjoey train configs/8head/sign_8head_128batch.yaml && python -m signjoey train configs/8head/sign_8head_256batch.yaml
echo batch 8 complete

echo batch 16
python -m signjoey train configs/16head/sign_16head_8batch.yaml && python -m signjoey train configs/16head/sign_16head_16batch.yaml && python -m signjoey train configs/16head/sign_16head_32batch.yaml && python -m signjoey train configs/16head/sign_16head_64batch.yaml && python -m signjoey train configs/16head/sign_16head_128batch.yaml && python -m signjoey train configs/16head/sign_16head_256batch.yaml
echo batch 16 complete

echo batch 24
python -m signjoey train configs/24head/sign_24head_8batch.yaml && python -m signjoey train configs/24head/sign_24head_16batch.yaml && python -m signjoey train configs/24head/sign_24head_32batch.yaml && python -m signjoey train configs/24head/sign_24head_64batch.yaml && python -m signjoey train configs/24head/sign_24head_128batch.yaml && python -m signjoey train configs/24head/sign_24head_256batch.yaml
echo batch 24 complete