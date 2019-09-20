#!/usr/bin/env python3

# Copyright 2019, Brno University of Technology (author: Karel Vesely)
# Apacha 2.0

import time

#wav_file="/mnt/matylda2/data/AMI/TS3008b/audio/TS3008b.Headset-2.wav"
#wav_file="/mnt/matylda5/iveselyk/NEUREM3/fisher_online/client/sample_ami_20s16k_sentence.wav" # Some 16khz file,
wav_file="snemovna_16_2018_06_27014.wav" # Some 16khz file,

DEBUG=False

# Read the wav,
from scipy.io import wavfile
fs, w = wavfile.read(wav_file)

# Open TCP socket to server,
import socket
TCP_IP = '127.0.0.1'
TCP_PORT = 5050
BUFFER_SIZE = 1024*16
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

# We will send data from background process,

# Send function simulating getting data from microphone,
def send_function(s,w):
    block=16000
    for ii in range(len(w)//block):
        chunk=w[ii*block:(ii+1)*block]
        if DEBUG: print("sending chunk: ", ii, chunk)
        s.send(chunk)
        time.sleep(1) # 1s
        #time.sleep(0.001) # 1s
    leftover_chunk=w[(len(w)//block)*block:]
    if len(leftover_chunk) > 0:
        if DEBUG: print("sending leftover chunk: ", ii)
        s.send(leftover_chunk)
    else:
        if DEBUG: print("no leftover chunk")

# Start the thread,
import threading
class sendWaveformThread(threading.Thread):
   def __init__(self, s, w):
      threading.Thread.__init__(self)
      self.threadID = 1
      self.args=(s,w)
   def run(self):
      if DEBUG: print ("Starting 'waveform-sender' thread")
      send_function(*self.args)
      if DEBUG: print ("Exiting 'waveform-sender' thread")
thr = sendWaveformThread(s,w)
thr.start()

# receive the output,
received_text=b''
while 1:
    if DEBUG: print("Waiting for output")
    data = s.recv(BUFFER_SIZE)
    if len(data) == 0: break
    received_text += data
    if DEBUG:
        print("Received: ", data) # show the 'binary' datagram we recieved,
    else:
        print(data.decode('utf8').lower(), end='') # now new-line,

# finally, store the output,
with open("text_output.txt", 'wb') as f:
    f.write(received_text)

