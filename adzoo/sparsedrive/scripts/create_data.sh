export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

#python tools/data_converter/B2D_converter.py nuscenes \
##    --root-path ./data/nuscenes \
#    --canbus ./data/nuscenes \
#    --out-dir ./data/infos/ \
#    --extra-tag nuscenes \
#    --version v1.0
python adzoo/sparsedrive/tools/data_converter/B2D_converter.py