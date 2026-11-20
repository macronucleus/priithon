
import os
import numpy as N
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# not working for Mac 20210209
# use pythonw and do it on command line (not on priithon)
# 20210324 upgraded matplotlib and works on priithon with some error

def saveImage(arr3d, outfn='test.mp4', fps=10, cmap='afmhot', vmin=None, vmax=None, frame_txt='z = %i'):
    """
    install ffmpeg by homebrew for mp4
    save in mp4 or gif
    use prepColorImg() for color image, use cmap=None

    cmap: jet, gray, afmhot, gist_heat, magma, etc...
    raibow is hsv
    """
    arr3d = arr3d[:,::-1] # flip y

    fig, ax, img = prepareFig(arr3d, cmap, vmin, vmax)

    if frame_txt:
        txt = ax.text(3, 7, frame_txt % 0) #'z = {:.1f} $\mu m$'.format(z[64]))
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Initialize the figure with an empty frame
    def init():
        img.set_data(np.zeros(arr3d.shape[1:3]))
        return img,

    # This function is called for each frame of the animation
    def animate(frame):   
        img.set_data(arr3d[frame])

        if frame_txt:
            txt.set_text(frame_txt % frame)
        return img,

    # Create the animation
    anim = animation.FuncAnimation(fig, animate, frames=len(arr3d),
                                   init_func=init, interval=20, blit=True)

    # Save the animation
    ext = os.path.splitext(outfn)[-1]
    if ext == '.mp4':
        myWriter = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
        anim.save(outfn, writer = myWriter)
    else:
        anim.save(outfn, writer = 'pillow', fps=fps)

    plt.close()

    return outfn

def prepareFig(arr3d, cmap='jet', vmin=None, vmax=None):
    ny, nx = arr3d.shape[1:3]
    ratio = 640 / nx
    
    fig, ax    = plt.subplots(nrows=1, ncols=1, figsize=(6.4, (ny*ratio)/100))

    ind = N.unravel_index(N.argmax(arr3d), arr3d.shape)
    z = ind[0]
    if vmax is None:
        vmax = arr3d.max()
    if vmin is None:
        vmin = arr3d.min()

    img = ax.imshow(arr3d[z], vmin=vmin, vmax=vmax, cmap=cmap)

    return fig, ax, img

def prepColorImg(arr, cmap='rg', colaxis=0, mis=(None,None), mas=(None,None)):
    assert arr.shape[colaxis] == len(cmap)
    if mis:
        assert arr.shape[colaxis] == len(mis)
    else:
        mis = [None for i in range(arr.shape[colaxis])]
    if mas:
        assert arr.shape[colaxis] == len(mas)
    else:
        mas = [None for i in range(arr.shape[colaxis])]
    assert arr.ndim <= 4

    if arr.ndim == 4:
        if colaxis == 1:
            brr = N.array([_prepColorImg(arr[:,i], mis[i], mas[i]) for i in range(arr.shape[1])])
        else:
            brr = N.array([_prepColorImg(a, mis[i], mas[i]) for i, a in enumerate(arr)])

    axisorder = 'rgb'
    rgb = N.zeros(brr[0].shape + (3,), dtype=N.uint8)
    for i, a in enumerate(brr):
        color = cmap[i].lower()
        axis = axisorder.index(color)
        rgb[...,axis] = a
    return rgb

def _prepColorImg(arr, mi=None, ma=None):
    if mi is None:
        mi = arr.min()
    if ma is None:
        ma = arr.max() - mi
    b = arr.astype(N.float32)
    b = N.clip(arr, mi, ma) - mi
    #b /= b.max()
    b = b / b.max()
    b *= 255
    return b.astype(N.uint8)



##### plot ------
def savePlot(x, y, outfn='plot.mp4', fps=10):
    """
    outfn: extension should be mp4 or gif
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def update(f):
        ax.cla() # ax をクリア
        ax.grid()
        ax.plot(x, y, c="gray")
        ax.plot(x[f], y[f], "o", c="red")

    #interval = fps/60 * 1000
    anim = animation.FuncAnimation(fig, update, frames=list(range(len(x))), interval=20)

    ext = os.path.splitext(outfn)[-1]
    if ext == '.mp4':
        myWriter = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
        anim.save(outfn, writer = myWriter)
    else:
        myWriter = 'imagemagick'
        anim.save(outfn, writer="pillow", fps=fps)
        
    plt.close()


#### no matlab
# https://stackoverflow.com/questions/5585872/python-image-frames-to-video
def frame2movie(img, outfn='', fps=10):
    from Priithon.all import U
    import os
    fns = 'test%03d.jpg'
    U.saveImg8_seq(img, fns)
    os.system("ffmpeg -f image2 -r %.2f -i %s -vcodec mpeg4 -y %s" % (fps, fns, outfn))
    for i in range(img.shape[0]):
        os.remove(fns % i)
    return outfn

def viewer2movie(viewer_id=0, zaxis=0, outfn='', intermediate_format='png', crf=17, fps=10):
    """
    use this function for consistency in LUT etc..

    crf: 16-18, 0 without any loss
    """
    import os, wx, subprocess
    from Priithon.all import Y
    fns = 'test%03d.' + intermediate_format

    v = Y.viewers[viewer_id]
    zs = v.zzslider[zaxis]
    nzs = zs.GetMax() + 1
    
    v.setSlider(0, zaxis)
    wx.Yield()
    
    for z in range(nzs):
        v.setSlider(z, zaxis)
        wx.Yield()
        Y.vSaveRGBviewport(viewer_id, fns % z, flipY=False)
    #U.saveImg8_seq(img, fns)
    #os.system("ffmpeg -f image2 -r %.2f -i %s -vcodec mpeg4 -y %s" % (fps, fns, outfn))
    os.system(
    #subprocess.run(
    'ffmpeg -framerate %.2f -i %s '
    '-c:v libx264 -crf %i -pix_fmt yuv420p -y %s'
    % (fps, fns, crf, outfn)
        )
    for z in range(nzs):
        os.remove(fns % z)
    return outfn



def frame2movie_cv2(img, outfn='', fps=10):
    import cv2
    #https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python


def addAxis(viewer_id=0, axis0='X', axis1='Z', startx=5, endx=20, starty=5, endy=20, color=(0,1,1), width=2, textsize=0.05, dist_arrow_txt=5):
    from Priithon.all import Y
    v = Y.viewers[viewer_id]
    if startx < 0:
        startx += v.data.shape[-1]
    if endx < 0:
        endx += v.data.shape[-1]
    if starty < 0:
        starty += v.data.shape[-2]
    if endy < 0:
        endy += v.data.shape[-2]
    
    ids = []
    idx = Y.vgAddArrows(viewer_id, N.array(([starty,startx],[starty,startx])),N.array(([starty,endx],[endy,startx])), color=color, width=width, refreshNow=False)
    ids.append(idx)

    idx = Y.vgAddTexts(viewer_id, [(endy+dist_arrow_txt,startx,axis1),(starty,endx+dist_arrow_txt,axis0)], size=textsize, color=color, width=width)
    ids.append(idx)

    return ids

def addScaleBar(viewer_id=0, startx=-30, endx=-5, posy=5, strlabel='1um', color=(0,1,1), width=2, textsize=0.05, dist_line_txt=5):
    from Priithon.all import Y
    ids = []
    v = Y.viewers[viewer_id]
    if startx < 0:
        startx += v.data.shape[-1]
    if endx < 0:
        endx += v.data.shape[-1]
    if posy < 0:
        posy += v.data.shape[-2]
        
    idx = Y.vgAddArrows(viewer_id, N.array(([posy, startx],)),N.array(([posy,endx],)), color=color, width=width, refreshNow=False)
    ids.append(idx)

    idx = Y.vgAddTexts(viewer_id, [(posy+dist_line_txt,(startx),strlabel)], size=textsize, color=color, width=width)
    ids.append(idx)

    return ids
