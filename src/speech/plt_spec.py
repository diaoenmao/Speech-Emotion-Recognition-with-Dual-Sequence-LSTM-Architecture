def specgram(self, x, NFFT=None, Fs=None, Fc=None, detrend=None,
                 window=None, noverlap=None,
                 cmap=None, xextent=None, pad_to=None, sides=None,
                 scale_by_freq=None, mode=None, scale=None,
                 vmin=None, vmax=None, **kwargs):
        """
        Plot a spectrogram.
        Compute and plot a spectrogram of data in *x*.  Data are split into
        *NFFT* length segments and the spectrum of each section is
        computed.  The windowing function *window* is applied to each
        segment, and the amount of overlap of each segment is
        specified with *noverlap*. The spectrogram is plotted as a colormap
        (using imshow).
        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data.
        %(Spectral)s
        %(PSD)s
        mode : {'default', 'psd', 'magnitude', 'angle', 'phase'}
            What sort of spectrum to use.  Default is 'psd', which takes the
            power spectral density.  'magnitude' returns the magnitude
            spectrum.  'angle' returns the phase spectrum without unwrapping.
            'phase' returns the phase spectrum with unwrapping.
        noverlap : int
            The number of points of overlap between blocks.  The
            default value is 128.
        scale : {'default', 'linear', 'dB'}
            The scaling of the values in the *spec*.  'linear' is no scaling.
            'dB' returns the values in dB scale.  When *mode* is 'psd',
            this is dB power (10 * log10).  Otherwise this is dB amplitude
            (20 * log10). 'default' is 'dB' if *mode* is 'psd' or
            'magnitude' and 'linear' otherwise.  This must be 'linear'
            if *mode* is 'angle' or 'phase'.
        Fc : int
            The center frequency of *x* (defaults to 0), which offsets
            the x extents of the plot to reflect the frequency range used
            when a signal is acquired and then filtered and downsampled to
            baseband.
        cmap
            A :class:`matplotlib.colors.Colormap` instance; if *None*, use
            default determined by rc
        xextent : *None* or (xmin, xmax)
            The image extent along the x-axis. The default sets *xmin* to the
            left border of the first bin (*spectrum* column) and *xmax* to the
            right border of the last bin. Note that for *noverlap>0* the width
            of the bins is smaller than those of the segments.
        **kwargs
            Additional kwargs are passed on to imshow which makes the
            specgram image.
        Returns
        -------
        spectrum : 2-D array
            Columns are the periodograms of successive segments.
        freqs : 1-D array
            The frequencies corresponding to the rows in *spectrum*.
        t : 1-D array
            The times corresponding to midpoints of segments (i.e., the columns
            in *spectrum*).
        im : instance of class :class:`~matplotlib.image.AxesImage`
            The image created by imshow containing the spectrogram
        See Also
        --------
        :func:`psd`
            :func:`psd` differs in the default overlap; in returning the mean
            of the segment periodograms; in not returning times; and in
            generating a line plot instead of colormap.
        :func:`magnitude_spectrum`
            A single spectrum, similar to having a single segment when *mode*
            is 'magnitude'. Plots a line instead of a colormap.
        :func:`angle_spectrum`
            A single spectrum, similar to having a single segment when *mode*
            is 'angle'. Plots a line instead of a colormap.
        :func:`phase_spectrum`
            A single spectrum, similar to having a single segment when *mode*
            is 'phase'. Plots a line instead of a colormap.
        Notes
        -----
        The parameters *detrend* and *scale_by_freq* do only apply when *mode*
        is set to 'psd'.
        """
        if NFFT is None:
            NFFT = 256  # same default as in mlab.specgram()
        if Fc is None:
            Fc = 0  # same default as in mlab._spectral_helper()
        if noverlap is None:
            noverlap = 128  # same default as in mlab.specgram()

        if mode == 'complex':
            raise ValueError('Cannot plot a complex specgram')

        if scale is None or scale == 'default':
            if mode in ['angle', 'phase']:
                scale = 'linear'
            else:
                scale = 'dB'
        elif mode in ['angle', 'phase'] and scale == 'dB':
            raise ValueError('Cannot use dB scale with angle or phase mode')

        spec, freqs, t = mlab.specgram(x=x, NFFT=NFFT, Fs=Fs,
                                       detrend=detrend, window=window,
                                       noverlap=noverlap, pad_to=pad_to,
                                       sides=sides,
                                       scale_by_freq=scale_by_freq,
                                       mode=mode)

        if scale == 'linear':
            Z = spec
        elif scale == 'dB':
            if mode is None or mode == 'default' or mode == 'psd':
                Z = 10. * np.log10(spec)
            else:
                Z = 20. * np.log10(spec)
        else:
            raise ValueError('Unknown scale %s', scale)

        Z = np.flipud(Z)

        if xextent is None:
            # padding is needed for first and last segment:
            pad_xextent = (NFFT-noverlap) / Fs / 2
            xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
        xmin, xmax = xextent
        freqs += Fc
        extent = xmin, xmax, freqs[0], freqs[-1]
        im = self.imshow(Z, cmap, extent=extent, vmin=vmin, vmax=vmax,
                         **kwargs)
        self.axis('auto')

        return spec, freqs, t, im