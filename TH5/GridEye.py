import serial
import sys
import struct
import numpy as np
import threading
from queue import Queue

import glob
from time import sleep
import ctypes
from numpy.ctypeslib import ndpointer

class GridEYEKit():
    def __init__(self):
        self._connected = False
        self.ser = serial.Serial()  # serial port object
        self.tarr_queue = Queue(1)
        self.thermistor_queue = Queue(1)
        self.multiplier_tarr = 0.25
        self.multiplier_th = 0.0625
        self._error = 0
         #Check start_update
        self.flag_check_stop = False  

        # if not self.connect():
        #     print "please connect Eval Kit"
        t = threading.Thread(target=self._connected_thread).start()

    def connect(self):
        """trys to open ports and look for valid data
        returns: true - connection good
        returns: False - not found / unsupported plattform
        """
        if self.ser.isOpen():
            self.ser.close()
        else:
            try:
                ports_available = self._list_serial_ports()
            except EnvironmentError:
                self._connected = False
                return False
            """try if kit is connected to com port"""
            for port in ports_available:
                self.ser = serial.Serial(port= port, baudrate= 115200,
                                         timeout=None)  # COM Port error is handled in list serial ports
                for i in range(5):
                    if self.serial_readline(bytes_timeout= 133):  # if 3 bytes identifyer found
                        self._connected = True
                        print("Connected\n")
                        return True  # GridEye found
                self.ser.close()
            self._connected = False
            return False

    def _list_serial_ports(self):
        """ This function is taken from Stackoverflow and will list all serial ports"""
        """Lists serial ports

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of available serial ports
        """
        if sys.platform.startswith('win'):
            ports = ['COM' + str(i + 1) for i in range(256)]

        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this is to exclude your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')

        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')

        else:
            raise EnvironmentError('Unsuppteorted platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result

    def _get_GridEye_data(self):
        """ get grid Eye data fron serial port and convert it to numpy array - also for further calculations"""
        array_pixel_8 = []
        thermistor = 0
        array_pixel_15 = []
        # print("HiHi")
        data = self.serial_readline()  # read grideye value
        if len(data) >= 131:
            for i in range(2, 130, 2):
                template_short = struct.unpack('<h', data[i:i + 2])[0]
                array_pixel_8.append(template_short)
            # array_pixel_15 = []
            array_pixel_15 = self.bAMG_PUB_IMG_LinearInterpolation(113, 113,array_pixel_8)
            array_pixel_15 = np.asarray(array_pixel_15)
            if len(array_pixel_15) >= 12769:
                self._error = 0
                if not data[1] & 0b00001000 == 0:  # Grid-Eye uses 12 bit signed data for calculating thermistor
                    data[1] &= 0b00000111
                    thermistor = -struct.unpack('<h', data[0:2])[0] * self.multiplier_th
                else:
                    thermistor = struct.unpack('<h', data[0:2])[0] * self.multiplier_th
        else:
            self._error = self._error + 1
            print( "Serial Fail")
        return thermistor, array_pixel_15   # thermistor: tuple , type : float/ tarr: mang hai chieu , type: float

    def _connected_thread(self):
        """" Background task reads Serial port and puts one value to queue"""
        while True:
            if self._connected == True:
                # print("HuHu")
                data = self._get_GridEye_data()# type data : tuple
                if self._connected == True:
                    if self.tarr_queue.full():
                        self.tarr_queue.get()
                        self.tarr_queue.put(data[1])
                    else:
                        self.tarr_queue.put(data[1])

                    if self.thermistor_queue.full():
                        self.thermistor_queue.get()
                        self.thermistor_queue.put(data[0])
                    else:
                        self.thermistor_queue.put(data[0])

                    if self._error > 5:
                        try:
                            self.ser.close()
                        except:
                            pass
                        self._connected = False
                        self._error = 0

    def get_thermistor(self):
        try:
            return self.thermistor_queue.get(True, 1)
        except:
            sleep(0.1)
            return 0

    def get_temperatures(self):
        try:
            return self.tarr_queue.get(True, 1)
        except:
            sleep(0.1)
            return np.zeros((113,113))

    def get_raw(self):
        try:
            return self.serial_readline()
        except:
            sleep(0.1)
            return np.zeros((113,113))

    def close(self):
        self._connected = False
        try:
            self.ser.close()
        except:
            pass

    def serial_readline(self, eol = '\r\n',start = '*',  bytes_timeout=133):
        """ in python 2.7 serial.readline is not able to handle special EOL strings - own implementation
        Returns byte array if EOL found in a message of max timeout_bytes byte
        Returns empty array with len 0 if not"""
        line = bytearray()
        eolByte = bytearray(eol, 'utf-8')
        startByte = bytes(start, 'utf-8')
        while True:
            # print("Chuan bi doc")
            c = self.ser.readline()
            if self.flag_check_stop == True:
                self.ser.close()
                # print(" Serial close")
                return []
            # print("Data: ", c)
            if c:
                line += c
                if line[:1] == startByte:
                    if line[-2:] == eolByte:
                        break
                    if len(line) > bytes_timeout:  # timeout
                        return []
                else:
                    line = bytearray()
            else:
                break
        return line[1:]
    def bAMG_PUB_IMG_LinearInterpolation(self,width: int, height: int, array_int: list)-> list:
        # print("Noi suy")
        lib = ctypes.CDLL('libfun.so')
        lib.bAMG_PUB_IMG_LinearInterpolation.argtypes = (ctypes.c_ubyte, ctypes.c_ubyte, ctypes.POINTER(ctypes.c_short))
        array_type_int = ctypes.c_short * len(array_int)
        lib.bAMG_PUB_IMG_LinearInterpolation.restype = ndpointer(dtype=ctypes.c_float, shape = (12769,))
        result = lib.bAMG_PUB_IMG_LinearInterpolation(ctypes.c_ubyte(width), ctypes.c_ubyte(height),array_type_int(*array_int))
        return result
