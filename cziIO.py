try:
    from . import generalIO
except ImportError:
    import generalIO

try:
    import czifile
    READABLE_FORMATS = ('czi',)
except ImportError:
    READABLE_FORMATS = ()
    
WRITABLE_FORMATS = ()

class CZIReader(generalIO.GeneralReader):
    def __init__(self, fn):
        """
        fn: file name
        """
        generalIO.GeneralReader.__init__(self, fn)

    def openFile(self):
        """
        open a file for reading
        """
        self.fp = czifile.CziFile(self.fn)
        self.handle = self.fp._fh
        
        self.readHeader()

    def readHeader(self):
        self.readMetaData()
        meta = self.metadata
        img = meta['ImageDocument']['Metadata']['Information']['Image']
        nx = img['SizeX']
        ny = img['SizeY']
        nw = img.get('SizeC', 1)
        nz = img.get('SizeZ', 1)
        if 'T' in self.fp.axes:
            nt = self.fp.shape[self.fp.axes.index('T')]
        else:
            nt = 1

        axes = self.fp.axes[:-3].replace('C', 'W').replace('B', '').replace('YX0', '')
        imgSeq = self.findImgSequence(axes)

        channels = img['Dimensions']['Channels']['Channel']
        waves = []
        self.exc = []
        for channel in channels:
            waves.append(channel['EmissionWavelength'])
            self.exc.append(channel['ExcitationWavelength'])

        self.setDim(nx, ny, nz=nz, nt=nt, nw=nw, dtype=self.fp.dtype, wave=waves, imgSequence=imgSeq)

        #pz = img['Dimensions']['Z']['Interval']['Increment']
        scl = meta['ImageDocument']['Metadata']['Scaling']['Items']['Distance']
        pdic = {}
        for s in scl:
            pdic['p%s' % s['Id'].lower()] = s['Value']*(10**6)

        self.setPixelSize(**pdic)

    def readMetaData(self):
        self.metadata = self.fp.metadata(False)

    def readSec(self, i):
        xy0 = self.roi_start[1:]
        xy1 = xy0 + self.roi_size[1:]

        sdir = self.fp.filtered_subblock_directory[i]
        subblock = sdir.data_segment()
        a = subblock.data(raw=False, resize=True, order=0)
        return a.ravel().reshape((self.ny, self.nx))[xy0[0]:xy1[0], xy0[1]:xy1[1]]
    
