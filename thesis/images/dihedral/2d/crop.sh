#!/bin/bash

INPUT_DIR="./"
OUTPUT_DIR="./cropped"

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.{png,jpg,jpeg}; do
    [ -e "$img" ] || continue

    filename=$(basename "$img")

    convert "$img" -crop 640x423+0+57 +repage "$OUTPUT_DIR/$filename"
done
