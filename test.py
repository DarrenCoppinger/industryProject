import unittest
import wave
import numpy as np

from main import inverter


class MyTestCase(unittest.TestCase):
    def test_inverter(self):
        # test the inverter function inverts the input wave
        # and that the sum of the input and output waves is zero
        CHUNK = 1024
        wf = wave.open("cafe_short.wav", 'r')
        file_data = wf.readframes(CHUNK)
        invert_data = inverter(file_data)

        file_data_int = np.frombuffer(file_data, dtype="int16")
        invert_data_int = np.frombuffer(invert_data, dtype="int16")

        mix_data = file_data_int + invert_data_int

        print("file_data_int= ", file_data_int)
        print("invert_data_int= ", invert_data_int)
        print("mix_data= ", mix_data)

        self.assertTrue(np.all((mix_data == 0)))


if __name__ == '__main__':
    unittest.main()
