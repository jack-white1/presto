#/usr/bin/env python

"""Inject a fake pulsar into real data, creating
a filterbank file.

Patrick Lazarus, June 26, 2012
"""
import sys
import optparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

import filterbank
import psr_utils

NUMSECS = 1.0 # Number of seconds of data to use to determine global scale
              # when repacking floating-point data into integers


def parse_gaussians(gaussian_strs):
    if not gaussian_strs:
        warnings.warn("Using default Gaussian profile (Amplitude=1.0 " \
                        "FWHM=0.05, and phase=0.0)")
        gaussians = [(1.0, 0.05, 0.0)]
    else:
        gaussians = []
        for gaussian_str in gaussian_strs:
            split = gaussian_str.split()
            if len(split) != 3:
                raise ValueError("Bad number of gaussian components " \
                        "should be 3, got %d" % len(split))
            amp = float(split[0])
            fwhm = float(split[1])
            phs = float(split[2])

            if fwhm <= 0 or fwhm > 1:
                raise ValueError("FWHM provided (%g) is bad, should be " \
                        "between 0 and 1" % fwhm)
            if phs < 0 or fwhm > 1:
                raise ValueError("Phase provided (%g) is bad, should be " \
                        "between 0 and 1" % phs)
            gaussians.append((amp,fwhm,phs))
    return gaussians


def main():
    print "%d input files provided" % len(args)
    for fn in args:
        print "Injecting pulsar signal into: %s" % fn
        fil = filterbank.FilterbankFile(fn)
        nbin = int(np.round(options.period/fil.tsamp)) # Number of bins 
                                                       # across the profile
        delays = psr_utils.delay_from_DM(options.dm, fil.frequencies)
        delay_bins = np.round(delays/fil.tsamp).astype(int)
    
        gaussians = parse_gaussians(options.gaussians)
        prof = np.zeros(nbin)
        for amp, fwhm, phs in gaussians:
            prof += amp*psr_utils.gaussian_profile(nbin, phs, fwhm)

        if options.dryrun:
            plt.figure()
            plt.plot(prof, 'k-')
            plt.show()
            sys.exit()

        profs = np.empty((nbin, fil.nchans))
        for ichan, bindelay in enumerate(delay_bins):
            profs[:,ichan] = psr_utils.rotate(prof, -bindelay)
        plt.imshow(profs.T, aspect='equal', origin='bottom', interpolation='nearest')
        plt.show()
        nprofs = fil.number_of_samples/nbin
        remainder = fil.number_of_samples % nbin
        for iprof in np.arange(nprofs):
            fil.samples[iprof*nbin:(iprof+1)*nbin] += profs
        fil.samples[nprofs*nbin:] += profs[:remainder]
        minimum = np.min(fil.samples[:int(NUMSECS/fil.tsamp)])
        fil.samples -= minimum
        median = np.median(fil.samples[:int(NUMSECS/fil.tsamp)])
        # Set median to 1/3 of dynamic range
        global_scale = (256.0/3.0) / median
        fil.samples *= global_scale
        fil.samples = np.clip(fil.samples, 0, 256)
        fil.dtype = 'uint8'
        fil.header['source_name'] += "_fake"
        outfn = options.outname % fil.header 
        print "Writing out file: %s" % outfn
        fil.write_filterbank_file(outfn) 


if __name__ == '__main__':
    parser = optparse.OptionParser(prog='injectpsr.py', \
                    version="v0.1 Patrick Lazarus (June 26, 2012)")
    parser.add_option("--dm", dest='dm', action='store', type='float', \
                    help="The DM of the (fake) injected pulsar signal. " \
                        "(This argument is required.", \
                    default=None)
    parser.add_option("-p", "--period", dest='period', action='store', \
                    default=None, type='float', \
                    help="The period (in seconds) of the (fake) injected " \
                        "pulsar signal. (This argument is required.)")
    parser.add_option("-g", "--gaussian", dest='gaussians', action='append', \
                    help="A string of 3 parameters defining a gaussian " \
                        "component to be injected. Be sure to quote the " \
                        "3 parameters together. The params are: 'amplitude " \
                        "FWHM phase'. Amplitude is not related to SNR in " \
                        "any way, FWHM and phase should be between 0 and 1. " \
                        "(Default: if no compoments are provided " \
                        "a Gaussian with amplitude=1.0, FWHM=0.05, and " \
                        "phase=0.0 will be used.)", \
                    default=[])
    parser.add_option("-n", "--dryrun", dest="dryrun", action="store_true", \
                    help="Show the pulse profile to be injected and exit. " \
                        "(Default: do not show profile, inject it)")
    parser.add_option("-o", "--outname", dest='outname', action='store', \
                    default="injected.fil", \
                    help="The name of the output file.")
    (options, args) = parser.parse_args()
    if options.period is None or options.dm is None:
        raise ValueError("Both a period and a DM _must_ be provided!")
    main()
