#Online tracking of vowel formants using LPC
##Patrick Callier
##2015

The idea here is just to use real-time mic input and LPC filtering, among other 
goodies, to explore vowel quality as a way of controlling a cursor in 2D. Current
problems include mapping the sort-of-kind-of trapezoidal or triangular vowel space to 
rectangular screen coordinates, and stabilizing the output of LPC to prevent the cursor 
jumping around too much.

get_lpc.py contains the main functionality. 

Depends on matplotlib, numpy, pyaudio, scipy, and scikits-talkbox