# Visualization

## draw-pixel-factors

**`draw-pixel-factors` visualizes the results of `pixel-decode`**

```bash
punkst draw-pixel-factors --in-tsv ${path}/pixel.decode.tsv --header-json ${path}/pixel.decode.json --in-color ${path}/color.rgb.tsv --out ${path}/pixel.png --scale 100 --xmin ${xmin} --xmax ${xmax} --ymin ${ymin} --ymax ${ymax}
```

`--in-tsv` specifies the input data file created by `pixel-decode`.

`--header-json` specifies the header created by `pixel-decode`.

`--in-color` specifies a tsv file with the colors for each factor. The first three columns will be interpreted as R, G, B values in the range $0-255$. The valid lines will be assigned to factors in the order they appear in this file.

`--xmin`, `--xmax`, `--ymin`, `--ymax` specify the range of the coordinates.

`--scale` scales input coordinates to pixels in the output image. `int((x-xmin)/scale)` equals the horizontal pixel coordinate in the image.

`--out` specifies the output png file.

If your specified `--transform` in `lda4hex`, one way to create the color table is to use the helper python script
```bash
python punkst/ext/py/color_helper.py --input ${path}/prefix.results.tsv --output ${path}/color
```