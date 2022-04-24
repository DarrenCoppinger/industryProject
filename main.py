# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import PyAudio
import sys

import pyaudio
import keyboard
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def inverter(data):
    # Convert byte string into integer
    wave = np.frombuffer(data, np.int16)

    # invert integer using NumPy
    invert_wave = np.invert(wave)

    # Convert integer back into byte string
    inverted_sound = np.frombuffer(invert_wave, np.byte)

    return inverted_sound


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    CHUNK = 1024
    WIDTH = 2
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    frames = []

    print("* recording")
    try:
        #     for i in range(0, int(RATE / CHUNK * sys.maxunicode)):
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            input_stream = stream.read(CHUNK)
            stream.write(input_stream, CHUNK)

            # Add to frame array
            frames.append(input_stream)
            # invert the wave
            inverted_stream = inverter(input_stream)
            print("i= ", i)
            if keyboard.is_pressed('x'):
                print("You pressed x")
                break
    except (KeyboardInterrupt, SystemExit):
        print('Exit Live noise cancelling')
    except Exception as e:
        print(str(e))

    print("* done")

    stream.stop_stream()
    stream.close()

    p.terminate()

    print("Live noise-cancelling mode")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
