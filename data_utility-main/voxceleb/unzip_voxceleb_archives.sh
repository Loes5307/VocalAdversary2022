#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$VOXCELEB_RAW_DIR" ]]; then
  echo "Please set VOXCELEB_RAW_DIR before calling this script"
  exit 1
fi

if [[ -z "$VOXCELEB_EXTRACT_DIR" ]]; then
  echo "Please set VOXCELEB_EXTRACT_DIR before calling this script"
  exit 2
fi

# make sure all potential directories exist
mkdir -p "$VOXCELEB_RAW_DIR" "$VOXCELEB_EXTRACT_DIR"

# general function for extracting
unzip_file () {
  ZIP_FILE=$1
  TARGET_PATH=$2
  if [ -f "$ZIP_FILE" ]; then
    echo "unzipping $ZIP_FILE"
    NUM_FILES=$(zipinfo -h "$ZIP_FILE" | grep -oiP '(?<=entries: )[[:digit:]]+')
    mkdir -p "$TARGET_PATH"
    unzip -o "$ZIP_FILE" -d "$TARGET_PATH" | tqdm --total "$NUM_FILES" >> /dev/null
  else
     echo "file $ZIP_FILE not exist."
  fi
}

# voxceleb1 train
VOX1_DEV_PATH="$VOXCELEB_RAW_DIR"/vox1_dev_wav.zip
VOX1_DEV_UNZIP_PATH="$VOXCELEB_EXTRACT_DIR"/voxceleb1/train
unzip_file "$VOX1_DEV_PATH" "$VOX1_DEV_UNZIP_PATH"

# voxceleb1 test
VOX1_TEST_PATH="$VOXCELEB_RAW_DIR"/vox1_test_wav.zip
VOX1_TEST_UNZIP_PATH="$VOXCELEB_EXTRACT_DIR"/voxceleb1/test
unzip_file "$VOX1_TEST_PATH" "$VOX1_TEST_UNZIP_PATH"

# voxceleb2 train
VOX2_DEV_PATH="$VOXCELEB_RAW_DIR"/vox2_dev_wav.zip
VOX2_DEV_UNZIP_PATH="$VOXCELEB_EXTRACT_DIR"/voxceleb2/train
unzip_file "$VOX2_DEV_PATH" "$VOX2_DEV_UNZIP_PATH"

# voxceleb2 test
VOX2_TEST_PATH="$VOXCELEB_RAW_DIR"/vox2_test_wav.zip
VOX2_TEST_UNZIP_PATH="$VOXCELEB_EXTRACT_DIR"/voxceleb2/test
unzip_file "$VOX2_TEST_PATH" "$VOX2_TEST_UNZIP_PATH"

