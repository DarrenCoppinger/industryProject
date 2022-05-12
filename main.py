import sys
import pyaudio
import keyboard
import numpy as np
import wave
import matplotlib.pyplot as plt
import noisereduce as nr

from scipy.io import wavfile

CHUNK = 1024
WIDTH = 2
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
FORMAT = pyaudio.paInt16

# Create pyaudio object
p = pyaudio.PyAudio()


def inverter(data):
    """
    this function generates the inverse of the provided wave stream chunk
    The NumPy libray function invert() is used to achieve this.
    :param data: takes a array of  stream bytes
    :return:
    """
    # Convert byte string into integer
    wave_data = np.frombuffer(data, np.int16)

    # invert integer using NumPy
    invert_wave = (np.invert(wave_data) + 1)

    # Convert integer back into byte string
    inverted_sound = np.frombuffer(invert_wave, np.byte)

    return inverted_sound


def live_anc():
    """
    create a wire (input channelled directly to output) to return
    the inverse of the sound detected by the device microphone

    :return: nothing
    """

    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    input_frames = []
    inverted_frames = []
    print("Live noise-cancelling mode")
    print("Press x to exit mode")
    try:
        while True:
            input_stream = stream.read(CHUNK)
            # stream.write(input_stream, CHUNK)

            # Add to frame array
            input_frames.append(input_stream)
            # invert the wave
            inverted_stream = inverter(input_stream)
            inverted_frames.append(inverted_stream)

            # output the inverted wave stream
            stream.write(inverted_stream, CHUNK)
            if keyboard.is_pressed('x'):
                print("You pressed x")
                break
    except (KeyboardInterrupt, SystemExit):
        print('Exit Live noise cancelling')
    except Exception as e:
        print(str(e))

    stream.stop_stream()
    stream.close()

    # create input wave file
    print("ANC input file output to ANC_input.wav")
    output_file("ANC_input.wav", input_frames)

    wf_input = wave.open("ANC_input.wav", 'rb')
    anc_input = wf_input.readframes(-1)
    anc_input_data = np.frombuffer(anc_input, dtype="int16")
    anc_input_data = anc_input_data/32768

    # create inverted wave file
    print("ANC inverted file output to ANC_inverted.wav ")
    output_file("ANC_inverted.wav", inverted_frames)

    wf_inverted = wave.open("ANC_inverted.wav", 'rb')
    anc_inverted = wf_inverted.readframes(-1)
    anc_inverted_data = np.frombuffer(anc_inverted, dtype="int16")
    anc_inverted_data = anc_inverted_data / 32768

    mix_data = anc_input_data + anc_inverted_data

    time = np.linspace(
        0,
        len(anc_input_data) / wf_inverted.getframerate(),
        num=len(anc_input_data)
    )
    # creates a new figure
    plt.figure(1)

    # title of the plot
    plt.title("Active Noise Cancelling")

    # label of x-axis
    plt.xlabel("Time")

    # label of y-axis
    plt.ylabel("Amplitude")

    #  plot
    plt.plot(time, anc_input_data, color="green", label="input")
    plt.plot(time, anc_inverted_data, color="blue", label="inverted")
    plt.plot(time, mix_data, color="red", label="mix")

    # add legend
    plt.legend()

    # save the plot
    plt.savefig('anc_output.png')

    # shows the plot
    plt.show()


def output_file(name, array):
    """
    Output a wav file with recorded data
    :param name: output filename
    :param array: data to be output to wav file
    :return:
    """
    wf = wave.open(name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(array))
    wf.close()


def fft_filter():
    """
    Apply Fast Fourier Transform analysis to
    :return:
    """
    dt_f = 0.001
    t_f = np.arange(0, 1, dt_f)

    print("Enter two frequency for the wave would you like to create: ")
    while True:
        try:
            input_freq_1 = int(input("Frequency 1: "))
            input_freq_2 = int(input("Frequency 2: "))
            break
        except ValueError:
            print("please enter a valid number")

    f = np.sin(2 * np.pi * input_freq_1 * t_f) + np.sin(2*np.pi*input_freq_2*t_f)
    f_clean = f
    f = f + 2.5*np.random.randn(len(t_f))

    n_f = len(t_f)
    fhat_f = np.fft.fft(f, n_f)
    PSD_f = fhat_f * np.conj(fhat_f) / n_f
    freq_f = (1 / (dt_f * n_f)) * np.arange(n_f)
    L_f = np.arange(1, np.floor(n_f / 2), dtype='int')

    # find frequencies with large powers
    indices_f = PSD_f > 100
    PSD_clean_f = PSD_f * indices_f
    fhat_f = indices_f * fhat_f
    # Inverse FFT for filtered signal
    ffilt_f = np.fft.ifft(fhat_f)

    plt.suptitle("Original v Noisy data")
    plt.plot(t_f, f, color='red', label='noisy data')
    plt.plot(t_f, f_clean, color='black', label='original data')
    plt.xlim(t_f[0], t_f[-1])
    plt.ylim(10, -10)
    plt.xlabel("time (sec)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    plt.suptitle("Original v Filtered data")
    plt.plot(t_f, np.real(f_clean), color='red', label='original data')
    plt.plot(t_f, np.real(ffilt_f), color='black', label="filtered data")
    plt.xlim(t_f[0], t_f[-1])
    plt.ylim(10, -10)
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(2, 1)
    plt.suptitle("FFT PSD Analysis")

    plt.sca(axs[0])
    plt.plot(np.real(freq_f[L_f]), np.real(PSD_f[L_f]), color='black', label='noisy data')
    plt.xlim(freq_f[L_f[0]], freq_f[L_f[-1]])
    plt.xlabel("Frequency Hz")
    plt.ylabel("PSD")
    plt.legend()

    plt.sca(axs[1])
    plt.plot(np.real(freq_f[L_f]), np.real(PSD_clean_f[L_f]), color='black', label='filtered data')
    plt.xlim(freq_f[L_f[0]], freq_f[L_f[-1]])
    plt.xlabel("Frequency Hz")
    plt.ylabel("PSD")
    plt.legend()

    plt.show()


# from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1: Np + 1] *= phases
    f[-1: -1 - Np: -1] = np.conj(f[1: Np + 1])
    return np.fft.ifft(f).real


# from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= int(min_freq), freqs <= int(max_freq))] = 1
    return fftnoise(f)


def record():
    # CHUNK_RECORD = 1024
    print("Record FORMAT =", FORMAT)
    print("Record CHANNELS =", CHANNELS)
    print("Record RATE =", RATE)
    print("Record CHUNK_RECORD =", CHUNK)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* start audio recording")
    print("stop recording by pressing x")

    record_frames = []
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            record_frames.append(data)
            if keyboard.is_pressed('x'):
                print("You pressed x")
                break
    except KeyboardInterrupt:
        print("Done recording")
    except Exception as e:
        print(str(e))

    print("* finish recording")

    output_filename = input("Enter output file name:")
    if output_filename[-4:] != ".wav":
        output_filename = output_filename + ".wav"

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(record_frames))
    wf.close()

    stream.stop_stream()
    stream.close()


def read_file(name):
    """
    read in an existing file
    :param name: the name of the file to be read
    :return: wf (waveform data), stream (stream data), dt (time delta), w_len (file length in seconds)
    """
    print("filename= ", name)
    # name = input("Enter WAV file name: ")
    while True:
        try:
            wf = wave.open(name, 'r')
            break
        except wave.Error:
            print("You must input a WAV audio file")
            name = input("Enter the name of the audio .wav file: ")
        except FileNotFoundError:
            print("The file name you have entered does not exist")
            name = input("Enter the name of the audio .wav file: ")

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    dt = 1 / wf.getframerate()
    w_len = (wf.getnframes() / wf.getframerate())
    print("input WIDTH= ", wf.getsampwidth())
    print("input CHANNELS= ", wf.getnchannels())
    print("input = FRAMERATE", wf.getframerate())
    print("dt= (sec)", dt)
    print("file length= (sec)", w_len)

    return wf, stream, dt, w_len


def plot_data(array1, array2, array3, length):
    print("dt= ", (1 / RATE) * CHUNK)
    print("length= ", length)
    print("n= ", (length / ((1 / RATE) * CHUNK)))

    t = np.arange(0, length, ((WIDTH / RATE) * CHUNK), dtype=object)
    print("t length= ", len(t))
    print("array1 length", len(array1))
    keyboard.read_key()

    # plot data
    plt.plot(t, array1, color='r', label="clean")

    plt.plot(t, array2, color='b', label="clean+noise")

    plt.plot(t, array3, color='g', label="noise")

    # x-axis label
    plt.xlabel("Time (sec)")
    # y-axis label
    plt.ylabel("Wave Amplitude")

    # graph title
    plt.suptitle("Wave Analysis")

    # include plot legend
    plt.legend()

    # show the plot
    plt.show()


def add_noise():
    noisy_frames = []
    output_data_frames = []
    # read in audio file
    orig_audio_name = input("Enter the name of the original audio .wav file: ")
    # orig_audio_name = "bush_fish.wav"
    (wavefile, stream, dt, w_len) = read_file(orig_audio_name)

    data = wavefile.readframes(-1)
    data_int = np.frombuffer(data, np.int16)

    # read in noise file
    noise_name = input("Enter the name of the noise audio .wav file: ")
    # noise_name = "cafe_mult.wav"
    (noise_wavefile, noise_stream, noise_dt, noise_w_len) = read_file(noise_name)

    noise_data = noise_wavefile.readframes(-1)
    noise_data_int = np.frombuffer(noise_data, np.int16)
    noise_data_int = noise_data_int[:wavefile.getnframes()]

    noise_data_int = noise_data_int

    print("data len", len(data_int))
    print("noise_data_int len", len(noise_data_int))
    print("data ", data_int[1])
    print("noise_data_int", noise_data_int[1])

    output_data = data_int + noise_data_int

    output_data_byte = np.frombuffer(output_data, np.byte)
    output_data_frames.append(output_data_byte)

    # gets the frame rate
    f_rate = wavefile.getframerate()

    # Plot x-axis in seconds
    time = np.linspace(
        0,
        len(data_int) / f_rate,
        num=len(data_int)
    )

    # matplotlib create a new plot
    plt.figure(1)

    # title of the plot
    plt.title("Input and Noise Signal Comparison")

    # label of x-axis
    plt.xlabel("Time")

    # label of y-axis
    plt.ylabel("Amplitude")

    plt.plot(time, noise_data_int, color='red', label='noise data')
    # plt.plot(noise_data_int, color='red', label='noisy')

    # original input plotting
    plt.plot(time, data_int, color='blue', label='input data')
    # plt.plot(data_int, color='blue', label='input')

    # include plot legend
    plt.legend()

    # shows the plot
    plt.show()

    # title of the plot
    plt.title("Combine input and noise signals")

    # label of x-axis
    plt.xlabel("Time")

    # label of y-axis
    plt.ylabel("Amplitude")

    # Output data plotting
    plt.plot(time, output_data, color='blue', label='input data')

    # include plot legend
    plt.legend()

    # shows the plot
    plt.show()

    output_filename = input("Enter new noisy file name:")
    if output_filename[-4:] != ".wav":
        output_filename = output_filename + ".wav"

    output_file(output_filename, output_data_frames)


def mult():
    mult_frames = []
    wavefile = wave.open("cafe_short.wav", 'r')

    data = wavefile.readframes(CHUNK)

    while data != b'':
        mult_frames.append(data)
        data = wavefile.readframes(CHUNK)

    wavefile = wave.open("cafe_short.wav", 'r')

    data = wavefile.readframes(CHUNK)

    while data != b'':
        mult_frames.append(data)
        data = wavefile.readframes(CHUNK)

    wavefile = wave.open("cafe_short.wav", 'r')

    data = wavefile.readframes(CHUNK)

    while data != b'':
        mult_frames.append(data)
        data = wavefile.readframes(CHUNK)

    wf = wave.open("cafe_mult.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(mult_frames))
    wf.close()


def noise_reduction():
    filename = input("Enter file name to perform noise cancelling on:")
    # wavefile = wave.open("bush_fish.wav", 'r')
    wavefile = wave.open(filename, 'r')

    data = wavefile.readframes(-1)
    data_int = np.frombuffer(data, np.int16)

    # Plot x-axis in seconds
    time = np.linspace(
        0,
        len(data_int) / wavefile.getframerate(),
        num=len(data_int)
    )

    # load data
    rate, data = wavfile.read("finaltest.wav")
    # title of the plot
    plt.title("Noisy v Filtered Signal")

    # label of x-axis
    plt.xlabel("Time")

    # label of y-axis
    plt.ylabel("Amplitude")
    plt.plot(time, data, color="red", label="noisy signal")

    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    plt.plot(time, reduced_noise, color="blue", label="filtered signal")

    # include plot legend
    plt.legend()

    plt.show()

    # title of the plot
    plt.title("Original v Filtered Signal")

    # label of x-axis
    plt.xlabel("Time")

    # label of y-axis
    plt.ylabel("Amplitude")

    plt.plot(time, data_int, color='red', label='original signal')
    # plt.plot(noise_data_int, color='red', label='noisy')

    # original input plotting
    plt.plot(time, reduced_noise, color='blue', label='filtered signal')
    # plt.plot(data_int, color='blue', label='input')

    # include plot legend
    plt.legend()

    # shows the plot
    plt.show()

    wavfile.write("reduced_noise.wav", rate, reduced_noise)


def play():
    play_name = input("Enter the name of the audio .wav file to be played: ")
    (wavefile, stream, dt, w_len) = read_file(play_name)

    data = wavefile.readframes(CHUNK)
    print("file reading")
    while data != b'':
        stream.write(data)
        data = wavefile.readframes(CHUNK)

    print("Close stream")
    stream.stop_stream()
    stream.close()


# user interface for programme
if __name__ == '__main__':
    print('#' * 80)
    print("Select programme mode:")
    while True:
        print('#' * 80)
        print("1: Play .wav audio file ")
        print("2: Record .wav audio file ")
        print("3: Add noise to .wav File ")
        print("4: Apply Noise Cancelling to .wav audio file")
        print("5: Live Active Noise Cancellation")
        print("6: FFT Demonstration")
        print("0: Exit Program")
        print('#' * 80)
        mode = input("Enter mode number: ")
        # mode = '3'
        if mode == '1':
            print("1: Play .wav audio file ")
            play()
        elif mode == '2':
            print("2: Record .wav audio file ")
            record()
            print("Play finished")
        elif mode == '3':
            print("3: Add noise to .wav File ")
            add_noise()
        elif mode == '4':
            print("4: Apply Noise Cancelling to .wav audio file")
            noise_reduction()
        elif mode == '5':
            print("5: Live Active Noise Cancellation")
            live_anc()
        elif mode == '6':
            print("6: FFT Demonstration")
            fft_filter()
        elif mode == '0':
            print("0: Exiting program")
            print('#' * 80)
            # Exit programme
            p.terminate()
            sys.exit()
        else:
            print("Please choose an option from the list")
