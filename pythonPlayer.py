#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import warnings
from scipy.io.wavfile import write

class ComplexExp(object):
    def __init__(self, k, N):
        assert N > 0, "N should be a nonnegative scalar"
        self.k = k
        self.N = N

        # Vector containing elements of time indexes
        self.n = np.arange(N)

        # Vector containing elements of the complex exponential
        self.exp_kN = np.exp(2j*cmath.pi*self.k*self.n / self.N)
        self.exp_kN *= 1 / (np.sqrt(N))

        # Vector containing real elements of the complex exponential
        self.exp_kN_real = self.exp_kN.real

        # Vector containing imaginary elements of the complex exponential
        self.exp_kN_imag = self.exp_kN.imag



# In[24]:


def q_11(N, k_list):
    assert isinstance(N, int), "N should be an integer"

    for k in k_list:
        # Creates complex exponential object with frequency k and duration N
        exp_k = ComplexExp(k, N)
        # Real and imaginary parts
        
        cpx_cos = exp_k.exp_kN_real
        cpx_sin = exp_k.exp_kN_imag
        # Plots real and imaginary parts
        
        cpx_plt = plt.figure()
        ax = cpx_plt.add_subplot(111)
        plt.stem(exp_k.n, cpx_cos, 'tab:blue', markerfmt='bo', label='Real part')
        plt.stem(exp_k.n, cpx_sin, 'tab:red', markerfmt='ro', label='Imaginary part')
        plt.title('Complex exponential: k = ' + str(k) + ', N = ' + str(N), fontsize=10)
        plt.xlabel('n', fontsize=9)
        plt.ylabel('x[n]', fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc = 1)
        # Aspect ratio credit: https://jdhao.github.io/2017/06/03/change-aspect-ratio-in-mpl/
        ratio = 1/(16/9)
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


# In[25]:


#Answer 1.1
list_of_ks = [0, 2, 9, 16]
duration_of_signal = 32
q_11(duration_of_signal, list_of_ks)


# In[28]:


#Question 1.2

def q_12(N, k_list):
    """
    Question 1.2: Equivalent complex exponentials. 
    Arguments:
        N: duration of the signal (int)
        k_list: frequency of the discrete complex exponential (list of ints)
    """
    assert isinstance(N, int), "N should be an integer"

    for k in k_list:
        # Creates complex exponential object with frequency k and duration N
        exp_k = ComplexExp(k, N)
        
        # Real and imaginary parts
        cpx_cos = exp_k.exp_kN_real
        cpx_sin = exp_k.exp_kN_imag
        
        # Plots real and imaginary parts
        cpx_plt = plt.figure()
        ax = cpx_plt.add_subplot(111)
        plt.stem(exp_k.n, cpx_cos, 'tab:blue', markerfmt='bo', label='Real part')
        plt.stem(exp_k.n, cpx_sin, 'tab:red', markerfmt='ro', label='Imaginary part')
        plt.title('Complex exponential: k = ' + str(k) + ', N = ' + str(N), fontsize=10)
        plt.xlabel('n', fontsize=9)
        plt.ylabel('x[n]', fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc = 1)
        # Aspect ratio credit: https://jdhao.github.io/2017/06/03/change-aspect-ratio-in-mpl/
        ratio = 1/(16/9)
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        plt.show()


# In[35]:


#Answer 1.2: Frequencies that are N apart
list_of_ks = [3, -29, 35]
duration_of_signal = 32
q_12(duration_of_signal, list_of_ks)


# In[33]:


#Question 1.3

def q_13(N, k_list):
    assert isinstance(N, int), "N should be an integer"

    for k in k_list:
        # Creates complex exponential object with frequency k and duration N
        exp_k = ComplexExp(k, N)
        # Real and imaginary parts
        cpx_cos = exp_k.exp_kN_real
        cpx_sin = exp_k.exp_kN_imag
        # Plots real and imaginary parts
        cpx_plt = plt.figure()
        ax = cpx_plt.add_subplot(111)
        plt.stem(exp_k.n, cpx_cos, 'tab:blue', markerfmt='bo', label='Real part')
        plt.stem(exp_k.n, cpx_sin, 'tab:red', markerfmt='ro', label='Imaginary part')
        plt.title('Complex exponential: k = ' + str(k) + ', N = ' + str(N), fontsize=10)
        plt.xlabel('n', fontsize=9)
        plt.ylabel('x[n]', fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc = 1)
        # Aspect ratio credit: https://jdhao.github.io/2017/06/03/change-aspect-ratio-in-mpl/
        ratio = 1/(16/9)
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        plt.show()


# In[34]:


# Answer 1.3: Opposite frequencies
list_of_ks = [-3, 3]
duration_of_signal = 32
q_13(duration_of_signal, list_of_ks)


# We see that the cosine wave is unaffected by the negative frequency, because cosine is an even function, whereas the sine wave is flipped since sine is an odd function. Although this isn't required in the report, I just thought it was interesting to point out.

# In[51]:


#Answer 1.4: More opposite frequencies
list_of_ks = [3, 29]
duration_of_signal = 32
q_13(duration_of_signal, list_of_ks)


# In[47]:


#Question 1.5

def q_15(N):
    """
    Question 1.5: Orthonormality 
    Arguments:
        N: duration of the signal (int)
    """
    assert isinstance(N, int), "N should be an integer"
    k_list = np.arange(N)
    l_list = np.arange(N)

    # Building a matrix with all signals
    cpx_exps = np.zeros((N,N), dtype=complex)
    for k in k_list:
        cpxexp = ComplexExp(k, N)
        cpx_exps[:, k] = cpxexp.exp_kN

    # Conjugate
    cpx_exps_conj = np.conjugate(cpx_exps)

    res = np.array(np.round(np.matmul(cpx_exps_conj, cpx_exps).real))
    print ("\n Matrix of inner products: Mp")
    print (res)
    fig, ax = plt.subplots()
    im = ax.imshow(res)
    plt.title('Inner products: N = ' + str(N), fontsize=10)
    plt.xlabel('l = [0, N - 1]', fontsize=9)
    plt.ylabel('k = [0, N - 1]', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()


# In[48]:


#Answer 1.5
duration_of_signal = 16
q15_mtx = q_15(duration_of_signal)


# When we perform matrix multiplication on two orthogonal matrices, we should get the identity matrix, as $q_n \cdot q_n = 1$ and $q_n \cdot q_m = 0 \ $ for   $n\neq m$. Since the elements of our set make up the elements of our matrices, we have an orthonormal set.

# In[49]:


#Answer 3.1

def cexpt(f, T, fs):
    assert T > 0, "Duration of the signal cannot be negative."
    assert fs != 0, "Sampling frequency cannot be zero"

    if fs < 0:
        warnings.warn("Sampling frequency is negative. Using the absolute value instead.")
        fs = - fs
    
    if f < 0:
        warnings.warn("Complex exponential frequency is negative. Using the absolute value instead.")
        f = -f

    # Duration of the discrete signal
    N = math.floor(T * fs)
    # Discrete frequency
    k = N * f / fs
    # Complex exponential
    cpxexp = ComplexExp(k, N)
    x = cpxexp.exp_kN
    x = np.sqrt(N) * x

    return x, N

def q_31(f, T, fs):
    # Retrieves complex exponential
    cpxexp, num_samples = cexpt(f, T, fs)
    # Cosine is the real part
    cpxcos = cpxexp.real



# In[58]:


#Question 3.2
def q_32(f0, T, fs):
    # Retrieves complex exponential
    cpxexp, num_samples = cexpt(f0, T, fs)
    # Cosine is the real part
    Anote = cpxexp.real
    # Playing the note
    write("Anote.wav", fs, Anote.astype(np.float32))


# In[59]:


#Answer 3.2
f0 = 440
T  = 2
fs = 44100
q_32(f0, T, fs)


# In[103]:


#Question 3.3
def q_33(note, T, fs):

    fi = 2**((note - 49) / 12)*440
    # Retrieves complex exponential
    cpxexp, num_samples = cexpt(fi, T, fs)
    # Cosine is the real part
    q33note = cpxexp.real
    # Playing the note
    write("q33note.wav", fs, q33note.astype(np.float32))

# If we pass the note name and note its number
def q_33_notename(note, T, fs):
    # mapping notes
    note_dict = ['G2', 'C3', 'C4', 'D4', 'D4S', 'E4', 'F4', 'G4', 'G4S', 'A4', 'A4S', 'B4', 'C5', 'D5', 'D5S', 'E5', 'F5', 'G5',
    'G5S', 'A5', 'A5S', 'B5', 'C6', 'C6S', 'D6', 'D6S', 'E6', 'F6', 'F6S', 'G6', 'A6', 'B6']

    note_fis = [23, 28, 40, 42, 43,
    44, 45, 47, 48, 49,
    50, 51, 52, 54,
    55, 56, 57, 59, 60,
    61, 62, 63, 64,
    65, 66, 67, 68, 69,
    70, 71, 73, 75]

    idx = note_dict.index(note)
    fi = note_fis[idx]

    # Retrieves complex exponential
    cpxexp, num_samples = cexpt(fi, T, fs)
    q33note = cpxexp.real
    # Playing the note
    file_name = str(note) + 'note.wav'
    write(file_name, fs, q33note.astype(np.float32))



# In[104]:


#Answer 3.3
note = 40
T = 2
fs = 44100
q_33(note, T, fs)


# In[137]:


#Question 3.4
def q_34(list_notes, list_times, fs):
    assert len(list_notes) == len(list_times), "List of musical notes and musical times should have same length"
    song = []
    for note, note_time in zip(list_notes, list_times):
        fi = 2**((note - 49) / 12)*440
        x, N = cexpt(fi, note_time, fs)
        song = np.append(song, x.real)
        song = np.append(song, np.zeros(10))

    # Writing song
    write("yorunikakeru1.wav", fs, song.astype(np.float32))



# In[138]:


#Answer 3.4
#mapping notes
G2=23
C3=28
C4=40
D4=42
D4S=43
E4=44
F4=45
G4=47
G4S=48
A4=49
A4S=50
B4=51
C5=52
D5=54
D5S=55
E5=56
F5=57
G5=59
G5S=60
A5=61
A5S=62
B5=63
C6=64
C6S=65
D6=66
D6S=67
E6=68
F6=69
F6S=70
G6=71
A6=73
B6=75

#Song
song_notes = [G4, A4, C5, G4S, G4, F4, D4S, F4, C5, A4S, C5, G4,
              F4, D4S, C4, D4S, F4, G4S, G4, G4, G5, F4, F5, G5, D4, D5, 
              D4S, D5S, D4, D5, A4S, C4, C5, A4S, G4, A4S, C5, G4S, G4, F4, 
              D5, C5, A4S, A4S, C5, D5, D5S, G4, F4, F4, D4S,
              C3, G2, C3, G3, C3, G2, C3, G2, A4S, C5, 
              D5S, A5S, C6, A5S, G5, C5, D5S, F5, G5, C5, D6S, D6, A5S, G5, A5S, C6, 
              A5S, G5, F5, C5, G5, F5, D5S, C5, B4, G5S, G5, D5, F5,
              D5S, A5S, G5S, G5, A4S, A4S, C5, D5S, A5S, G5, F5, D5S, C5, D5S, F5, 
              G5, C5, D6S, D6, A5S, G5, A5S]
rhythm=6/13
b = 1*rhythm
h = 0.5*rhythm
q = 0.25*rhythm;

song_times = [h, h, h, h, h, h, h, h, h, h, q, h,
              h, b, h, h, h, h, h, h, h, h, h, h, h, h, 
              h, h, h, h, h, h, h, h, h, h, h, h, h, h, 
              h, h, h, h, h, h, h, b, h, h, h,
              q, q, b, b, h, h, q, q, q, q,
              q, q, q, h, q, h, h, h, h, q, q, q, q, q, h, q,
              h, h, h, h, h, h, q, h, q, h, h, h, h,
              h, h, h, q, q, h, h, h, h, q, q, q, q, q, h, 
              q, h, h, h, h, q, q]
q_34(song_notes, song_times, fs)

