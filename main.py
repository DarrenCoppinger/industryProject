# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import pyaudio
import keyboard
import numpy as np
import wave
import matplotlib.pyplot as plt

import struct

# CHUNK = 1024
CHUNK = 2
WIDTH = 2
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5

# Create pyaudio object
p = pyaudio.PyAudio()

# Output variables
WAVE_OUTPUT_FILENAME = "output.wav"
FORMAT = pyaudio.paInt16

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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
    filename = input("Enter WAV file name: ")
    (wavefile, stream, dt, w_len) = read_file(filename)
    print("Reading WAV file")
    print("wavefile channels", wavefile.getnchannels())
    # plt.plot(wavefile)

    # read first byte CHUNK of wav file
    file_data = wavefile.readframes(CHUNK)

    # create matrix of wave data
    input_array = []
    noise_array = []
    input_plus_noise_array = []

    output_frames = []
    inverted_array = []

    # calculate time series
    t = np.arange(0, w_len, (dt*CHUNK))

    # read in byte from wave file
    while file_data != b'':
        try:
            # Play audio from file
            # stream.write(file_data)
            # print("file_data=", file_data)

            # covert byte data to integer data
            # input_data_int = np.frombuffer(file_data, np.int16)[0]
            # input_data_int = np.frombuffer(file_data, np.int16)
            # input_data_int = int.from_bytes(file_data, sys.byteorder)
            input_data_int = np.frombuffer(file_data, np.int16)

            # covert int data back to byte data
            # input_data_byte = input_data_int.to_bytes(CHUNK, sys.byteorder)
            # input_data_byte = np.frombuffer(input_data_int, np.byte)
            # input_data_byte = struct.pack(input_data_int)

            # # stream.write(input_data_byte)

            # input_data = np.frombuffer(file_data, np.int16)
            # print("input_data= ", input_data)
            # add integer data to array
            input_array.append(input_data_int)

            # create noise
            noise = np.random.randint(-1000, 1000)
            # output noise data to array
            noise_array.append(noise)
            # print("noise= ", noise)

            # add noise to input audio
            input_plus_noise = input_data_int + noise

            # output input and noise data to array
            input_plus_noise_array.append(input_plus_noise)

            # convert from integer back to bytes
            input_plus_noise_byte = np.frombuffer(input_plus_noise, np.byte)
            stream.write(input_plus_noise_byte)

            # # input_plus_noise_bytes = input_plus_noise.tobytes()
            # input_plus_noise_bytes = np.frombuffer(input_plus_noise, np.byte)
            # input_plus_noise_wave = np.frombuffer(input_plus_noise, np.byte)

            # output byte data to array
            output_frames.append(input_plus_noise_byte)


            # print("input_plus_noise= ", input_plus_noise)
            # int_input_plus_noise = np.frombuffer(input_plus_noise, np.int16)[0]

            # print("int_input_plus_noise= ", int_input_plus_noise)
            # input_plus_noise_array.append(int_input_plus_noise)

            # input_plus_noise_data = np.frombuffer(input_plus_noise, np.int16)[0]

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
    # plot_data(input_array, input_plus_noise_array, noise_array, t)

    # create output file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(wavefile.getnchannels())
    wf.setsampwidth(wavefile.getsampwidth())
    wf.setframerate(wavefile.getframerate())
    wf.writeframes(b''.join(output_frames))
    wf.close()


    # plot input plus noise wave
    # plot_data(input_array, t, "noisy")

    # Terminate PyAudio object
    p.terminate()
    # Exit programme
    sys.exit()


def read_file(name):
    try:
        # wf = wave.open(name, 'r')
        wf = wave.open("gettysburg10.wav", 'r')
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

    dt = 1/wf.getframerate()
    w_len = (wf.getnframes()/wf.getframerate())
    print("dt= (sec)", dt)
    print("file length= (sec)", w_len)

    return wf, stream, dt, w_len


def plot_data(array1, array2, array3, t):
    # t = np.arange(0, length, (dt*CHUNK))

    # plot data
    plt.plot(t, array1, color='r', label="clean")

    plt.plot(t, array2, color='b', label="clean+noise")

    plt.plot(t, array3, color='g', label="noise")

    # x-axis label
    plt.xlabel("Time (sec)")
    # y-axis label
    plt.ylabel("Wave Amplitude")

    # graph title
    plt.suptitle("Wave")

    # include plot legend
    plt.legend()

    # show the plot
    plt.show()


def add_noise(f, t):
    noise = 2.5*np.random.randn()
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
    mode = input("Enter mode number: ")
    if mode == '1':
        print("1: Live Active Noise Cancellation")
        live_anc()
    elif mode == '2':
        print("2: WAV File Noise Cancelling")
        file_anc()
    print('#' * 80)



