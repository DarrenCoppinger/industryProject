# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import pyaudio
import keyboard
import numpy as np
import wave
import matplotlib.pyplot as plt

CHUNK = 1024
# CHUNK = 1
WIDTH = 2
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
FORMAT = pyaudio.paInt16

# Create pyaudio object
p = pyaudio.PyAudio()


def inverter(data):
    # Convert byte string into integer
    wave_data = np.frombuffer(data, np.int16)

    # invert integer using NumPy
    invert_wave = np.invert(wave_data)

    # Convert integer back into byte string
    inverted_sound = np.frombuffer(invert_wave, np.byte)

    return inverted_sound


def live_anc():
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    frames = []

    try:
        #     for i in range(0, int(RATE / CHUNK * sys.maxunicode)):
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            input_stream = stream.read(CHUNK)
            # stream.write(input_stream, CHUNK)

            # Add to frame array
            frames.append(input_stream)
            # invert the wave
            inverted_stream = inverter(input_stream)

            # output the inverted wave stream
            stream.write(inverted_stream, CHUNK)
            print("i= ", i)
            if keyboard.is_pressed('x'):
                print("You pressed x")
                break
    except (KeyboardInterrupt, SystemExit):
        print('Exit Live noise cancelling')
    except Exception as e:
        print(str(e))

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Live noise-cancelling mode")


def file_anc():
    # Read in WAV file

    (wavefile, stream, dt, w_len) = read_file()
    print("Reading WAV file")

    # read first byte CHUNK of wav file
    file_data = wavefile.readframes(CHUNK)

    # create matrix of wave data
    input_array = []
    noise_array = []
    input_plus_noise_array = []
    output_frames = []
    inverted_array = []

    # calculate time series
    # t = np.arange(0, w_len, (dt*CHUNK))

    # read in byte from wave file
    while file_data != b'':
        try:
            # Play audio from file
            # stream.write(file_data)

            # covert byte data to integer data
            input_data_int = np.frombuffer(file_data, np.int16)
            # print("input_data_int length = ", len(input_data_int))

            # print("input_data_int= ", input_data_int)

            # test = np.frombuffer(input_data_int, np.byte)
            # stream.write(test)

            # add integer data to array
            input_array.append(input_data_int[0])
            # # input_array.append(input_data_int[0])
            # keyboard.read_key()
            # i = 0
            # while i < len(input_data_int):
            #     input_array.append(input_data_int[i])
            #     i = i + 1
            # if i == 1:
            #     print("input_data_int[i]= ", input_data_int[i])

            # create noise
            # Note on len(input_data_int). Last CHUNK might not be a full CHUNK.
            noise = np.random.randint(-1000, 1000, len(input_data_int))
            # print("noise length = ", len(noise))
            # noise = np.random.randint(-1000, 1000)
            # #noise = np.random.randint(-1000, 1000, CHUNK)

            # print("noise= ", noise)

            # output noise data to array
            noise_array.append(noise[0])

            # add noise to input audio
            input_plus_noise = input_data_int + noise
            # print("input_plus_noise= ", input_plus_noise)

            # output input and noise data to array
            input_plus_noise_array.append(input_plus_noise[0])

            # convert from integer back to bytes
            input_plus_noise_byte = np.frombuffer(input_plus_noise, np.byte)

            # Play input audio with added noise
            # stream.write(input_plus_noise_byte)

            # output byte data to array
            output_frames.append(input_plus_noise_byte)

            # Read next chuck of data
            file_data = wavefile.readframes(CHUNK)

            if keyboard.is_pressed('x'):
                print("You pressed x")
                break
        except (KeyboardInterrupt, SystemExit):
            print('Exit File active noise cancelling')
            break
        except Exception as e:
            print(str(e))

    # Close stream after finishing reading data
    stream.stop_stream()
    stream.close()

    print("End of WAV file")

    # plot input wave
    plot_data(input_array, input_plus_noise_array, noise_array, w_len)

    # create output file
    wf = wave.open("output.wav", 'wb')
    # # wf.setnchannels(wavefile.getnchannels())
    # # wf.setsampwidth(wavefile.getsampwidth())
    # # wf.setframerate(wavefile.getframerate())
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(output_frames))
    wf.close()

    # plot input plus noise wave
    # plot_data(input_array, t, "noisy")

    # Terminate PyAudio object
    p.terminate()
    # Exit programme
    sys.exit()


def fft_filter():
    print("fft")
    # Read in WAV file
    (wavefile, stream, dt, w_len) = read_file()
    print("Reading WAV file")

    new_CHUNK = 1
    # read first byte CHUNK of wav file
    file_data = wavefile.readframes(new_CHUNK)

    # create matrix of wave data
    fft_input_array = []
    fft_noise_array = []
    fft_input_plus_noise_array = []
    fft_output_frames = []
    inverted_array = []

    # read in byte from wave file
    while file_data != b'':
        try:
            # covert byte data to integer data
            # input_data_int = np.frombuffer(file_data, np.int16)
            input_data_int = np.frombuffer(file_data, dtype=np.int16)

            # add integer data to array
            fft_input_array.append(input_data_int[0])

            # create noise
            # Note on len(input_data_int). Last CHUNK might not be a full CHUNK.
            # noise = np.random.randint(-1000, 1000, len(input_data_int))
            noise = 5000*np.random.randn(len(input_data_int))

            # output noise data to array
            fft_noise_array.append(noise[0])

            # add noise to input audio
            input_plus_noise = input_data_int + noise

            # output input and noise data to array
            fft_input_plus_noise_array.append(input_plus_noise[0])

            # convert from integer back to bytes
            input_plus_noise_byte = np.frombuffer(input_plus_noise, np.byte)

            # output byte data to array
            fft_output_frames.append(input_plus_noise_byte)

            # Read next chuck of data
            file_data = wavefile.readframes(new_CHUNK)

            if keyboard.is_pressed('x'):
                print("You pressed x")
                break
        except (KeyboardInterrupt, SystemExit):
            print('Exit File active noise cancelling')
            break
        except Exception as e:
            print(str(e))

    # Close stream after finishing reading data
    stream.stop_stream()
    stream.close()

    print("End of WAV file")

    # # plot input wave
    t = np.arange(0, w_len, ((1 / RATE) * new_CHUNK))

    n = len(fft_input_plus_noise_array)
    fhat = np.fft.fft(fft_input_plus_noise_array, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n)
    L = np.arange(1, np.floor(n / 2), dtype='int')

    n_input = len(fft_input_array)
    fhat_input = np.fft.fft(fft_input_array, n_input)
    PSD_input = fhat * np.conj(fhat_input) / n_input
    freq_input = (1 / (dt * n_input)) * np.arange(n_input)
    L_input = np.arange(1, np.floor(n_input / 2), dtype='int')

    # find frequencies with large powers
    indices = PSD > 1
    PSD_clean = PSD * indices
    fhat = indices * fhat
    # Inverse FFT for filtered signal
    ffilt = np.fft.ifft(fhat)

    # fig, axs = plt.subplots(3, 1)
    #
    # print("w_len= ", w_len)
    # print("WIDTH= ", WIDTH)
    # print("RATE= ", RATE)
    # print("t= ", len(t))
    # print("fft_input_array= ", len(fft_input_array))
    # print("fft_input_plus_noise_array= ", len(fft_input_plus_noise_array))
    #
    # plt.sca(axs[0])
    # plt.plot(t, fft_input_plus_noise_array, color='k', label='noisy')
    # plt.plot(t, fft_input_array, color='c', label='clean')
    # # plt.xlim(2, 2.2)
    # plt.legend()
    #
    # plt.sca(axs[1])
    # plt.plot(t, ffilt, color='k', label="filtered")
    # plt.xlim(2, 2.2)
    # plt.legend()
    #
    # plt.sca(axs[2])
    #
    # plt.plot(freq[L], PSD[L], color='k', label='input + noise')
    # plt.plot(freq_input[L_input], PSD_input[L_input], color='c', label='input')
    # # plt.plot(freq[L], PSD_clean[L], color='k', label='Filtered')
    # plt.xlim(freq[L[0]], freq[L[-1]])
    # # plt.xlim(freq[L[0]], 1000)
    # # plt.ylim(0, 2)
    # plt.legend()
    #
    # plt.show()


    dt_f = 0.001
    t_f = np.arange(0, 1, dt_f)
    f = np.sin(2*np.pi*50*t_f) + np.sin(2*np.pi*120*t_f)
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

    fig, axs = plt.subplots(4, 1)

    plt.sca(axs[0])
    plt.plot(t_f, f, color='k', label='noisy')
    plt.plot(t_f, f_clean, color='c', label='clean')
    plt.xlim(t_f[0], t_f[-1])
    plt.legend()

    plt.sca(axs[1])
    plt.plot(t_f, ffilt_f, color='k', label="filtered")
    plt.xlim(t_f[0], t_f[-1])
    plt.legend()

    plt.sca(axs[2])
    plt.plot(freq_f[L_f], PSD_f[L_f], color='k', label='noisy')
    plt.plot(freq_f[L_f], PSD_clean_f[L_f], color='c', label='filtered')
    # plt.plot(freq[L], PSD_clean[L], color='k', label='Filtered')
    plt.xlim(freq_f[L_f[0]], freq_f[L_f[-1]])
    # plt.xlim(freq[L[0]], 1000)
    # plt.ylim(0, 2)
    plt.legend()

    plt.sca(axs[3])
    plt.plot(freq_f[L_f], PSD_clean_f[L_f], color='c', label='filtered')
    plt.xlim(freq_f[L_f[0]], freq_f[L_f[-1]])
    plt.legend()

    plt.show()

    # create output file
    print("FFT Output RATE= ", RATE)
    wf_output = wave.open("FFT_OUTPUT.wav", 'wb')
    wf_output.setnchannels(wavefile.getnchannels())
    wf_output.setsampwidth(wavefile.getsampwidth())
    wf_output.setframerate(wavefile.getframerate())
    wf_output.writeframes(b''.join(fft_output_frames))
    wf_output.close()

    # Terminate PyAudio object
    p.terminate()
    # Exit programme
    sys.exit()


def fft_simple_filter():
    print("fft_simple_filter")
    # Read in WAV file

    dt_f = 0.001
    t_f = np.arange(0, 1, dt_f)
    f = np.sin(2*np.pi*50*t_f) + np.sin(2*np.pi*120*t_f)
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

    fig, axs = plt.subplots(4, 1)

    plt.sca(axs[0])
    plt.plot(t_f, f, color='k', label='noisy')
    plt.plot(t_f, f_clean, color='c', label='clean')
    plt.xlim(t_f[0], t_f[-1])
    plt.legend()

    plt.sca(axs[1])
    plt.plot(t_f, ffilt_f, color='k', label="filtered")
    plt.xlim(t_f[0], t_f[-1])
    plt.legend()

    plt.sca(axs[2])
    plt.plot(freq_f[L_f], PSD_f[L_f], color='k', label='noisy')
    plt.plot(freq_f[L_f], PSD_clean_f[L_f], color='c', label='filtered')
    # plt.plot(freq[L], PSD_clean[L], color='k', label='Filtered')
    plt.xlim(freq_f[L_f[0]], freq_f[L_f[-1]])
    # plt.xlim(freq[L[0]], 1000)
    # plt.ylim(0, 2)
    plt.legend()

    plt.sca(axs[3])
    plt.plot(freq_f[L_f], PSD_clean_f[L_f], color='c', label='filtered')
    plt.xlim(freq_f[L_f[0]], freq_f[L_f[-1]])
    plt.legend()

    plt.show()


    # Terminate PyAudio object
    p.terminate()
    # Exit programme
    sys.exit()


# from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)


def record():
    CHUNK_RECORD = 1024
    print("Record FORMAT =", FORMAT)
    print("Record CHANNELS =", CHANNELS)
    print("Record RATE =", RATE)
    print("Record CHUNK_RECORD =", CHUNK_RECORD)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_RECORD)

    print("* start audio recording")
    print("stop recording by pressing x")

    record_frames = []
    try:
        while True:
            # for i in range(0, int(RATE / CHUNK_RECORD * RECORD_SECONDS)):
            # while True:
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
    p.terminate()


def read_file():
    # name = input("Enter WAV file name: ")
    try:
        # wf = wave.open(name, 'r')
        wf = wave.open('darren.wav', 'r')
    except wave.Error:
        print("You must input a WAV audio file")
        sys.exit()
    except FileNotFoundError:
        print("The file name you have entered does not exist")
        sys.exit()

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
    # keyboard.read_key()

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


def add_noise(f, t):
    noise = 2.5 * np.random.randn()
    f_noise = f + noise
    print("f_noise= ", f_noise)
    f_return = np.frombuffer(f_noise, np.int16)[0]
    print("f_return= ", f_return)
    return f_return


if __name__ == '__main__':
    print('#' * 80)
    print("Select programme mode:")
    print("1: Live Active Noise Cancellation")
    print("2: WAV File Active Noise Cancelling")
    print("3: Play WAV File ")
    print("5: Record audio .wav File ")
    # mode = input("Enter mode number: ")
    mode = '3'
    if mode == '1':
        print("1: Live Active Noise Cancellation")
        live_anc()
    elif mode == '2':
        print("2: WAV File Noise Cancelling")
        file_anc()
    elif mode == '3':
        print("2: FFT")
        fft_filter()
    elif mode == '4':
        print("4: ")
    elif mode == '5':
        print("5: Record")
        record()

    print('#' * 80)
