# imgio
imgio is a unified interface with diverse image file formats for microscopy.

# Dependency
-----------------------------
* `numpy`
* `pillow`
* `tifffile <= 2021.7.2` (optional for reading/writing multipage tiff/ImageJ file/ome.tif)
* `nd2 <= 0.5.3` (optional for reading nd2) # under development
* `czifile <= 2019.7.2`  (optional for reading czi) # under development
* `oiffile <= 2024.5.24` (optional for reading oif) # under development
* `readlif <= 0.6.5` (optional for reading lif)  # under development
  (GPL license)

Below packages are to use BioFormats (slow, and GPL license if you choose to use it)
* `javabridge` (optional for reading more image file formats)
* `python-bioformats` (optional for reading more image file formats, see License)
* `lxml` (optional for reading more image file formats)
* `Java Development Kit` (optional for reading more image file formats)
