#!/bin/bash
# Run ON the lukas machine. Pulls Ideogram-4 + FLUX.2-dev weights from the cluster.
set -e
rsync -avP --partial \
  sarim.hashmi@10.67.33.23:/shared/home/sarim.hashmi/usenix/generators/ideogram-4-fp8 \
  sarim.hashmi@10.67.33.23:/shared/home/sarim.hashmi/usenix/generators/FLUX.2-dev \
  /home/lukas/users/shashmi/liars-dividend-generation/
echo "done:"
du -sh /home/lukas/users/shashmi/liars-dividend-generation/{ideogram-4-fp8,FLUX.2-dev}
