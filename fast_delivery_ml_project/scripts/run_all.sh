#!/usr/bin/env bash
set -e
python src/eda.py --config config.yaml
python src/preprocess.py --config config.yaml
python src/train_baselines.py --config config.yaml --task sentiment
python src/train_baselines.py --config config.yaml --task rating
# Uncomment to run BERT (GPU recommended)
# python src/train_bert.py --config config.yaml --task sentiment
# python src/train_bert.py --config config.yaml --task rating
echo "All done. Check figures/ and experiments/"
