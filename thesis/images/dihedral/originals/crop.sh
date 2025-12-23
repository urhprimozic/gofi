#!/bin/bash

INPUT_DIR="./"
OUTPUT_DIR="cropped"
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

HEADER_HEIGHT=250

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/Dn_1dim_n*.png; do
    [ -e "$img" ] || continue

    filename=$(basename "$img")

    # Dn_1dim_n7.png -> 7
    n=$(echo "$filename" | sed -E 's/.*_n([0-9]+)\.png/\1/')

    convert "$img" \
        -fill white -draw "rectangle 0,0 1920,170" \
        -font "$FONT" -fill black -pointsize 60 \
        -gravity North -annotate +0+80 "n=$n" \
        "$OUTPUT_DIR/$filename"
done

