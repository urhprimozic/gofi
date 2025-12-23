#!/bin/bash

INPUT_DIR="./"
OUTPUT_DIR="renamed"
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
#!/bin/bash

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/prob_of_convergence_D_*.png; do
    [ -e "$img" ] || continue

    filename=$(basename "$img")

    # prob_of_convergence_D_3.png -> 3
    n=$(echo "$filename" | sed -E 's/.*_D_([0-9]+)\.png/\1/')

    convert "$img" \
        \( -size 640x57 xc:white \) -gravity North -composite \
        -font "$FONT" -fill black -pointsize 28 \
        -gravity North -annotate +0+20 "n=$n" \
        "$OUTPUT_DIR/$filename"
done

