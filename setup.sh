#!/bin/bash

mkdir -p build/dataset
mkdir -p build/model
mkdir -p build/backup

DATA_FILE=build/dataset/data.cfg
TRAIN_FILE=build/dataset/train.txt
VALID_FILE=build/dataset/valid.txt
NAMES_FILE=build/dataset/names.txt
IMAGES=build/images
MODEL=build/model
BACKUP=build/backup

LABEL=face
MODEL_FILE=dataset/${LABEL}.cfg

function setup_darknet() {
  git clone https://github.com/AlexeyAB/darknet build/darknet
  # GPU
  cmake -B build/darknet_out -S build/darknet -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  ln -s build/darknet_out/compile_commands.json .
  cmake --build build/darknet_out --target darknet
}

function generate_dataset() {
  tar xf dataset/images.tar.bz2 -C build
  realpath ${IMAGES}/*.jpg > input.txt
  lines=$(wc -l input.txt | awk '{print $1}')
  TRAIN=$((lines * 9 / 10))
  VALID=$((lines - TRAIN))
  head -n $TRAIN input.txt > ${TRAIN_FILE}
  tail -n $VALID input.txt > ${VALID_FILE}
  echo ${LABEL} > ${NAMES_FILE}
  rm input.txt

  cat << EOF > ${DATA_FILE}
classes = 1
train = ${TRAIN_FILE}
valid = ${VALID_FILE}
names = ${NAMES_FILE}
backup = ${BACKUP}
EOF
}

function model_convert() {
  python3 tools/quantification/yolo_to_h5.py ${MODEL_FILE} ${BACKUP}/${LABEL}_last.weights ${MODEL}/${LABEL}.h5
  python3 tools/quantification/h5_to_pb.py ${MODEL}/${LABEL}.h5 ${MODEL} ${LABEL}.pb
  python3 tools/quantification/pb_to_tflite.py ${IMAGES} ${MODEL}/${LABEL}.pb ${MODEL}/${LABEL}.tflite
  cp ${BACKUP}/${LABEL}_last.weights ${MODEL}/${LABEL}.weights
  
}
function darknet_train() {
  ./build/darknet_out/darknet detector train ${DATA_FILE} ${MODEL_FILE}
  model_convert
}

