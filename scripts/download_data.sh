#/bin/bash

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
wget https://s3.amazonaws.com/fair-data/parlai/projects/talkthewalk/talkthewalk.tgz
tar -xzvf talkthewalk.tgz
rm talkthewalk.tgz
