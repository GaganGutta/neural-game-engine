#!/usr/bin/env bash
# Full pipeline, data to playable model. Pass a config as $1 (default: small).
#
#   bash scripts/reproduce.sh configs/small.yaml   # CPU, a few hours
#   bash scripts/reproduce.sh configs/full.yaml    # one GPU, a couple of days
set -euo pipefail

CFG="${1:-configs/small.yaml}"
py() { python "$@"; }

frames=$(python -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(c['data']['frames'])")
workers=$(python -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(c['data'].get('workers',4))")
root=$(python -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(c['data']['root'])")
env=$(python -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(c['data']['env'])")

echo "==> 1/6  collect $frames frames from $env"
py -m ngx.data.collect --out "$root" --env "$env" --frames "$frames" --workers "$workers"

echo "==> 2/6  train tokenizer"
py -m ngx.train.train_vqvae --config "$CFG"

echo "==> 3/6  tokenize dataset"
py -m ngx.data.tokenize --config "$CFG"

echo "==> 4/6  train dynamics"
py -m ngx.train.train_dynamics --config "$CFG"

echo "==> 5/6  benchmark"
py -m ngx.eval.bench --config "$CFG"

echo "==> 6/6  drift + demo gif"
py -m ngx.eval.drift --config "$CFG"
py scripts/make_gif.py --config "$CFG"

echo
echo "done. play it:  python play.py --config $CFG"
