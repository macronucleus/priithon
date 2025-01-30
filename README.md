# Priithon
Priithon is a Python platform for image analysis and image analysis algorithm development.

While the focus here is on algorithm development, it makes it very easy to develop applications that would not require any programming skills, i.e. making "end user" biologist's or astronomer's applications.

# Installation
-----------------------------
Dependencies  
* `numpy`
* `scipy`
* `wxPython` (optional for GUI)
* `PyOpenGL` (optional for GUI)
* `wxmplot` (optional for plotting on Apple silicon) # under development
* `pyFFTW` (optional for fast Fourier transform) 
  GPL License (see License.md, even if  you don't install it, you can still use Fourier transform using numpy)
* `imgio` (optional for reading file formats other than the dv/mrc format)

To install, use conda environments in `envs` directory (contains pyFFTW).  
To create a new environment:  
`$ conda create -n [env_name] --file env_name.txt`  
To add to the existing environment:  
`$ conda install --name [env_name] --file env_name.txt`  

Alternatively, you can install the dependencies for your platform manually like:  
`$ conda create -n [env_name] -c conda-forge numpy scipy wxpython
pyopengl wxmplot pillow 
[pyfftw, tifffile, nd2, czifile, oiffile, readlif]`  

After activating the environment by typing  
`$ conda activate [env_name]`  
Install imgio:  
`$ pip install git+http://github.com/macronucleus/imgio@main`  
Then install Priithon:  
`$ pip install git+http://github.com/macronucleus/priithon@master`  

After installation, type on the command line:  
`$ priithon`  

On the priithon shell, type the following commands to test if the
installation was done successfully:  
`>>> Y.test()`  
`>>> Y.test2()`  
If these commands does not work, then it is likely that wxpython
version does not match even though the error message (appears on the
command prompt in the back of the priithon shell) suggests some OpenGL
problems.  
Make sure to use the newest wxpython version from conda-forge or PIP.  

# Usage
-----------------------------
Drag and drop your image file and select `View` to show
multi-dimensional images, or `load and assign to var` to assign the
image array to a python variable.  

If you have assigned your image array to `a` for example, then you
can:  
`>>> Y.view(a) # view as gray scale`  
`>>> b = Y.vd(0) # obtain image in the viewer as variable "b"`
`>>> Y.view2(a) # view as multicolor`  
`>>> a.header.pxlsize # shows pixel size of your image data`
`>>> af = F.sfft(a) # full Fourier transform with origin corrected`   
`>>> a = F.isfft(af) # full inverse Fourer transform with origin
corrected`  
`>>> af = F.rsfft(a) # real Fourier transform with origin corrected`  
`>>> a = F.irsfft(af) # real inverse Fourer transform with origin
corrected`  
`>>> Y.ploty([0,1,2,3])`

# Short-cut keys for viewer
-----------------------------
- **a**: show /amplitudes/ of complex data set

- **c**: cycle through many /color map/ modes:
  - gray scale
  - gray scale /logarithmic/
  - rainbow red to blue
  - /circular/ rainbow - good for phase images, where -PI is the same as +PI
  - "heat like" black body
  - "viridis"
  - "magma"
  - fast cycling rainbow with 0 black - good to see iso-conturs and
    gradients
	
- **ctl (command) + c**: copy the image appeared on the viewer to the
clipboad

- **f**: open new viewer showing the ("half-shaped",real-) fft(2d) of
  the image
  
- **F**: open new viewer showing the inverse fft(2d) of the current
  image (should be a "half-shaped" fft image)
  
- **g**: cycle through: no /grid/, one pixel grid, ten pixel grid
  
- **h**: auto histogram

- **o**:  cycle through 4 /origin/ modes:
    - (0,0) at left bottom
	- (0,0) at left top
	- (0,0) at center ( for fft images )
	- (0,0) at center &amp; "double width" ( for rfft images)

- **p**: show /phases/ of complex data set

- **v**: open new viewer showing "maximum intensity projection" (/"vomit"/) along z-axis

- **x**: open new viewer showing x-z side on view

- **y**: open new viewer showing y-z side on view

- **0**: reset zoom to one pixel per data point and move image to left bottom in viewer frame

- **9**: center image in viewer frame d "double" zoom - zoom in

- **page up/down**: zoom in/out l toggle histogram with and with out /logarithmic/ /y/-axis

- **Home**:

- **arrow keys left,right**: walk through /z/ axis (or what ever the higher dimensions are for)

- **arrow keys up, down**: walk through /t/ axis (or what ever the higher dimensions are for)

- **arrow keys with <shift> OR <control>**: shift image by quarter of it's size in the respective direction

- **Drag + <alt>**: Translate image

# History
Priithon was developmed by [Sebastian Hasse]
(https://github.com/sebhaase/priithon) until around 2010.   
Since then, I have continued to use it but only the functions most
often used by me have been maintained.   
