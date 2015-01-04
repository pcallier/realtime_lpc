#!/usr/bin/env python
""" 
Estimate formants using LPC for streaming mic input.

I stole much of this from stack overflow:
http://stackoverflow.com/questions/25107806/estimate-formants-using-lpc-in-python
As well as from an answer to: http://stackoverflow.com/questions/892199/detect-record-audio-in-python


"""

import sys
import time
import wave
import math
import itertools
import wave
import signal
import multiprocessing

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sys import byteorder
from struct import pack
from multiprocessing.managers import SyncManager

import numpy
import pyaudio
from scipy.signal import lfilter, hamming, resample
from scikits.talkbox import lpc

THRESHOLD = 500
CHUNK_SIZE = 2048
FORMAT = pyaudio.paInt16
NPFORMAT = numpy.int16
RATE = 44100

def wav_as_numpy(file_path):
    # Read from file.
    spf = wave.open(file_path, 'r') # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav

    # Get file as numpy array.
    x = spf.readframes(-1)
    x = numpy.fromstring(x, 'Int16')
    
    return x


def get_formants(x, Fs):

    #for e in x:
    #    print >> sys.stderr, e
    
    # Get Hamming window.
    N = len(x)
    w = numpy.hamming(N)

    # Apply window and high pass (pre-emphasis) filter.
    x1 = x * w
    x1 = lfilter([1.], [1., 0.63], x1)
    
    # Resample to make estimates better??
    new_Fs = 22050
    new_N = numpy.floor((float(N) * float(new_Fs))/Fs)
    #print new_N
    x1 = resample(x1, new_N, window=None)
    Fs = int(new_Fs)

    # Get LPC.
    ncoeff = 0 + Fs / 1000
    A, e, k = lpc(x1, ncoeff)

    try:
        # Get roots.
        rts = numpy.roots(A)
        rts = [r for r in rts if numpy.imag(r) >= 0]

        # Get angles.
        angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))

        # Get frequencies.
        frqs = angz * (Fs / (2 * math.pi))
        frq_indices = numpy.argsort(frqs)
        frqs = [frqs[i] for i in frq_indices]
        bws = [-1/2*(Fs/(2*numpy.pi))*numpy.log(numpy.abs(rts[i])) for i in frq_indices]
        frqs = [freq for freq, bw in itertools.izip(frqs, bws) if freq > 90 and bw < 400]
    except numpy.linalg.LinAlgError:
        frqs = []

    return frqs
    
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = numpy.empty([0], dtype=NPFORMAT)
    for i in snd_data:
        r = numpy.append(r, int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = numpy.empty([0], dtype=NPFORMAT)

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r = numpy.append(r, i)

            elif snd_started:
                r = numpy.append(r, i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data = snd_data[::-1]       # reverse
    snd_data = _trim(snd_data)
    snd_data = snd_data[::-1]
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    assert seconds != 0
    r = numpy.array([0 for i in xrange(int(seconds*RATE))], dtype=NPFORMAT)
    r = numpy.append(r, snd_data)
    r = numpy.append(r, [0 for i in xrange(int(seconds*RATE))])
    return r


def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = numpy.empty([0], dtype=NPFORMAT)


    while 1:
        # little endian, signed short
        #snd_data = array('h', stream.read(CHUNK_SIZE))
        new_data = stream.read(CHUNK_SIZE)
    
        #print "Data: \"%s\"" % new_data
        try:
            snd_data = numpy.array(numpy.fromstring(new_data,dtype=numpy.int16),dtype=NPFORMAT)
        except ValueError as e:
            snd_data = numpy.array([0],dtype=NPFORMAT)
    
        if byteorder == 'big':
            snd_data = snd_data.byteswap()
        r = numpy.append(r, snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            print "recording"
            snd_started = True

        if snd_started and num_silent > 30:
            print "done recording"
            break
        
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def get_audio(q):
    """
    Infinite loop streaming audio from microphone. Puts audio data to a
    queue, q. Be careful that q is shareable across threads, see:
    http://stackoverflow.com/questions/3217002/how-do-you-pass-a-queue-reference-to-a-function-managed-by-pool-map-async
    """
    
    p = pyaudio.PyAudio()
    sample_width = p.get_sample_size(FORMAT)
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=False,
        frames_per_buffer=CHUNK_SIZE)
    keep_looping = True
    
    try:
        while keep_looping:
            # little endian, signed short
            new_data = stream.read(CHUNK_SIZE)

            try:
                snd_data = numpy.array(numpy.fromstring(new_data,dtype=numpy.int16),dtype=NPFORMAT)
            except ValueError as e:
                snd_data = numpy.array([0],dtype=NPFORMAT)
        
            if byteorder == 'big':
                snd_data = snd_data.byteswap()
            
            # put tuples into q of sample width, sample chunk
            q.put((sample_width, snd_data))
    except KeyboardInterrupt:    
        print >> sys.stderr, "Exiting get_audio loop..."
    finally:
        stream.close()
        p.terminate()
        print >> sys.stderr, "Cleaned up get_audio objects..."


def mgr_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    


if __name__ == '__main__':
    # set up animation
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.set_ylim(0, 1000)
    ax.set_xlim(700,2000)
    vowel_point = ax.plot(1200, 500, 'ro')
    f1 = 500
    f2 = 200
    formants_list = []
#    plt.show()

    # some multiprocessing setup
    thread_manager = SyncManager()
    thread_manager.start(mgr_init)
    audio_queue = thread_manager.Queue()
    
    def vowel_update(framedata):
        sw, snd_data = audio_queue.get()
        # get formants
        if is_silent(snd_data) != True:
            formants = get_formants(snd_data, RATE)
            print "F1: {f1:10.3f}\tF2: {f2:10.3f}\tF3: {f3:10.3f}".format(f1=formants[0], f2=formants[1], f3=formants[2])
            formants_list.extend(formants)
        
            if len(formants_list) > 10:
                formants = zip ( formants_list )
                formants = numpy.mean(formants, axis = 1)
                formants_list = []
            
                if len(formants) >= 3:
                    vowel_point.set_xdata(formants[1])
                    vowel_point.set_ydata(formants[0])
        else:
            formants_list=[]
        return vowel_point,

    print "Starting..."
    ani = animation.FuncAnimation(fig, vowel_update, blit=False, repeat=True)
    print "No reallly..."
    
    # start audio input thread
    audio_input = multiprocessing.Process(target=get_audio, args=(audio_queue, ))
    audio_input.daemon=True
    audio_input.start()
    try:
        plt.show()

    finally:
        # wait for children to shutdown
        while audio_input.exitcode== None:
            time.sleep(0.1)
        print >> sys.stderr, "Shutting down..."
        thread_manager.shutdown()

        
    
    