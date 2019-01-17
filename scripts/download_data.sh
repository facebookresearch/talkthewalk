#/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

if [ -z "$1" ]
  then
  data_dir='./data'
else
  data_dir=$1
fi

echo "Downloading data to $data_dir"
if [ ! -d "$data_dir" ]; then
  mkdir ${data_dir}
fi
cd ${data_dir}
wget https://dl.fbaipublicfiles.com/parlai/projects/talkthewalk/talkthewalk.tgz
tar -xzvf talkthewalk.tgz
rm talkthewalk.tgz
