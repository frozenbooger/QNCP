import pyvisa
from pyvisa.constants import Parity,StopBits
import ctypes
import os, sys, pathlib
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from scipy.signal import find_peaks
import inspect
import vxi11
import time
import socket
import json
import threading
import ast

def robust(method, *arg):
    def robust_method(self, *arg):
        try:
            result = method(self,*arg)
        except:
            self.dev.close()
            time.sleep(1)
            self.dev = self.rm.open_resource(self.address)
            result = method(self,*arg)
        return result
        
    robust_method.__name__ = method.__name__
    robust_method.__doc__ = method.__doc__
    robust_method.__module__ = method.__module__
    return robust_method

#================================================================
# Spectrum Analyzer - Rigol_DSA800 series 
#================================================================

class Rigol_DSA800:
    def __init__(self,address): #'TCPIP::<IP ADDRESS>::INSTR' 
        self.address = address            
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.dev.write_termination='\n'
        self.dev.read_termination='\n'

    @robust    
    def center_freq(self,*f):  # center frequency
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:FREQuency:CENTer {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:FREQuency:CENTer?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq

    @robust
    def span_freq(self,*f):  # frequency range 
        if bool(f) == True: 
            __f = f[0] 
            self.dev.write(':SENSe:FREQuency:SPAN {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:FREQuency:SPAN?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq

    @robust
    def rbw(self,*f):  # resolution bandwidth
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:BANDwidth:RESolution {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:BANDwidth:RESolution?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq
    @robust    
    def vbw(self,*f):  # video bandwidth
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:BANDwidth:VIDeo {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:BANDwidth:VIDeo?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq
        
    @robust
    def __Hz(self,f):  # in Hz, support unit. Default: MHz
        if type(f) == str:
            if re.search('[mM]',f) != None:
                return 1e6*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[kK]',f) != None:
                return 1e3*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[hH]',f) != None:
                return 1*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[gG]',f) != None:
                return 1e9*float(re.sub('[a-zA-Z]','',f))
        else: # float, or str that contains only numbers
            return float(f)*1e6

    @robust
    def auto(self): # auto VBW and RBW
        self.dev.write(':SENSe:BANDwidth:RESolution:AUTO ON')
        self.dev.write(':SENSe:BANDwidth:VIDeo:AUTO ON')
        
    @robust
    def set_frequency_bounds(self,f_lower,f_upper):
        self.dev.write('SENS:FREQ:STARt {}'.format(f_lower))
        self.dev.write('SENS:FREQ:STOP {}'.format(f_upper))        
        
    @robust
    def monitor(self): ##print spectral signal, can also detect peaks
        
        ##ask for freq limits for plotting
        
        f_lower = float(self.dev.query(':SENS:FREQ:STARt?'))
        f_upper = float(self.dev.query(':SENS:FREQ:STOP?'))
        
        ##initiate
        
        self.dev.write(':INITiate:CONTinuous OFF')
        self.dev.write(':TRACe1:MODE WRITe')
        self.dev.write(':FORMat[:TRACe][:DATA] ASCii')
        
        ##trigger 
        
        self.dev.write(':INITiate')
        while (int(self.dev.query(':STATus:OPERation:CONDition?'))&(1<<3)):
            pass
        
        ##collect and turn into array
        
        rawdata=self.dev.query(':TRACe:DATA? TRACe1',)
        data = rawdata.split(", ")
        data[0] = data[0].split()[1]
        data = np.array([float(i)for i in data])
        freq = np.linspace(f_lower,f_upper,601) #601 is the sample size
        
        ##find peaks and (using peak prominences)
        
        # peaks, properties = find_peaks(data, prominence=1,width=20)
        
        return freq, data
    
    @robust
    def detect_peaks(self): #find peaks

        ##initiate
        self.dev.write(':INITiate:CONTinuous OFF')
        self.dev.write(':TRACe1:MODE WRITe')
        self.dev.write(':TRACe:MATH:PEAK:TABLe:STATe ON')
        self.dev.write(':FORMat[:TRACe][:DATA] ASCii')
        
        ##trigger 
        self.dev.write(':INITiate')
        while (int(self.dev.query(':STATus:OPERation:CONDition?'))&(1<<3)):
            pass
        
        rawpeaks = self.dev.query(':TRAC:MATH:PEAK?')
        peaks = np.array(rawpeaks.split(",")).astype(np.float)
        freqs=peaks[::2]
        amps=peaks[1::2]
        
        return freqs,amps
    
#===========================================================================
# Oscilloscope - Rigol DS1000E series 
#===========================================================================            
class Rigol_DS1000E:
    def __init__(self,address): # 'TCPIP::<IP ADDRESS>::INSTR'
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()

    @robust
    def meas(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    @robust
    def run(self):
        self.dev.write(':RUN')
    @robust
    def stop(self):
        self.dev.write(':STOP')
    @robust
    def single(self):
        self.dev.write(':SINGle')
    
    @robust
    def config(self,ch,cfg):  # specifically designed for Pete lock error monitor
        if type(ch) == str:  # channel info, default ch2
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
            
        # pete lock error monitor
        if re.search('(pete)',cfg,re.IGNORECASE) != None:
            __tPID = 120e-6    # PID response time in Pete
            __tDiv = 10e-6    # second/div
            __tOffset = __tPID + 12*__tDiv    # offset
            __vTrig = 0    # trigger level
            __vScale = 0.5    # 500mV/div, 
            self.run()
            self.dev.write(':TIMebase:SCALe {}'.format(__tDiv))   # time scale 10us/div
            self.dev.write(':TIMebase:MAIN:OFFSet {}'.format(__tOffset))            
            self.dev.write(':CHAN{}:COUPling DC'.format(__ch))  # DC coupling
            self.dev.write(':CHAN{}:SCALe {}'.format(__ch,__vScale)) 
            self.dev.write(':CHAN{}:OFFSet 0'.format(__ch))  # DC coupling
            self.dev.write(':TRIG:MODE EDGE')  # trigger on edge
            self.dev.write(':TRIG:SWEep AUTO')  # force trigger when not triggered
            self.dev.write(':TRIG:EDGe:SOURce CHAN{}'.format(__ch))  #trigger
            self.dev.write(':TRIG:EDGe:LEV {}'.format(__vTrig))      

#===========================================================================
# Oscilloscope DS1000z Series
#===========================================================================            
class Rigol_DS1000z:
    def __init__(self,address):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()
        
    @robust
    def reset(self):
        self.dev.write('*RST')

    @robust
    def measure(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    
    @robust
    def run(self):
        self.dev.write(':RUN')
        
    @robust
    def stop(self):
        self.dev.write(':STOP')
        
    @robust
    def single(self):
        self.dev.write(':SINGle')
    
    @robust
    def screenshot(self,ch):
        """ 
        Description: The screenshot function acquires the time and voltage information
        from the oscilliscope at the oscilliscopes current settings

        Input: ch : channel : integer = {1,2,3,4}
        Output: time_data : array which has timing information : np.array
                volt_data : array which has voltage infromation : np.array
        Example: 
        >> acq1.screenshot(ch1)

        np.array([0,1,2,3,4,...,99,100]),np.array([0,1,2,3,2,...,2,3])
        """
        self.dev.write(":WAV:SOUR CHAN{}".format(ch))
        self.dev.write(":WAV:MODE NORMal")
        self.dev.write(":WAV:FORM ASC")
        self.dev.write(":WAV:DATA? CHAN{}".format(ch))

        rawdata = self.dev.read_raw()
        rawdata = rawdata.decode('UTF-8')
        volt_data = rawdata[11:] #removes header and ending of data
        volt_data = np.array([float(volt_data) for volt_data in volt_data.split(',')])

        t = float(self.dev.query(':WAVeform:XINCrement?'))
        time_data = np.arange(0,t*len(volt_data),t)

        return time_data, volt_data
    
    @robust
    def scale_offset(self, ch, scale, offset):
        """ 
        Description: The scale_offset function sets the scale and offset parameters of an 
        oscilliscope channel to the scale of choice 

        Input: ch : channel : integer = {1,2,3,4}
               scale : usually in the order of your signal/8 : float
               offset : usually a multiple of the scale : float
               
        Output: None : none : None
        Example: 
        >> acq1.scale_offset(ch, (max_volt-min_volt)/8, -(max_volt-min_volt)/4)
        """
        self.dev.write(':CHANnel{}:SCAL {}'.format(ch,scale))
        self.dev.write(':CHANnel{}:OFFS {}'.format(ch,offset))
        
    @robust
    def time_scale_offset(self, scale, offset):
        """ 
        Description: The scale_offset function sets the timing scale and offset parameters of an 
        oscilliscope 

        Input: ch : channel : integer = {1,2,3,4}
               scale : usually in the order of your signal width/4 : float
               offset : usually a multiple of the scale : float
               
        Output: None : none : None
        Example: 
        >> acq1.scale_offset(ch, 1e-6/4, 0)

        """
        self.dev.write('TIMebase:MAIN:SCAL {}'.format(scale))
        self.dev.write('TIMebase:MAIN:OFFS {}'.format(offset))
        
    @robust
    def channel_state(self, ch, state):
        self.dev.write(':CHANnel{}:DISPlay {}'.format(ch,state))
        
    @robust
    def trigger_set(self, ch, edge, level):
        """ 
        Description: Set trigger settings and level

        Input: ch : channel : int = {1,2,3,4}
               edge : edge to trigger on : int = {'POS'=1, 'NEG'=2, 'RFAL'=3}
               level : level to trigger at : float 
        Output: None : None 

        Example: 
        >>trigger_set(1,0,3)

        """
        edges = ['POS','NEG','RFAL']
        self.dev.write(':TRIGger:EDGE:SOURce CHANnel{}'.format(ch))
        self.dev.write(':TRIGger:EDGE:SLOPe {}'.format(edges[edge]))
        self.dev.write(':TRIGger:EDGE:LEVel {}'.format(level))

#===========================================================================
# Oscilloscope - Rigol DMO5000
#=========================================================================== 

class Rigol_DMO5000:
    def __init__(self,address):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()
        
    @robust
    def reset(self):
        self.dev.write('*RST')

    @robust
    def measure(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    
    @robust
    def run(self):
        self.dev.write(':RUN')
        
    @robust
    def stop(self):
        self.dev.write(':STOP')
        
    @robust
    def single(self):
        self.dev.write(':SINGle')
    
    @robust
    def screenshot(self,ch):
        """ 
        Description: The screenshot function acquires the time and voltage information
        from the oscilliscope at the oscilliscopes current settings

        Input: ch : channel : integer = {1,2,3,4}
        Output: time_data : array which has timing information : np.array
                volt_data : array which has voltage infromation : np.array
        Example: 
        >> acq1.screenshot(ch1)

        np.array([0,1,2,3,4,...,99,100]),np.array([0,1,2,3,2,...,2,3])
        """
        self.dev.write(":WAV:SOUR CHAN{}".format(ch))
        self.dev.write(":WAV:MODE NORMal")
        self.dev.write(":WAV:FORM ASC")
        self.dev.write(":WAV:DATA? CHAN{}".format(ch))

        rawdata = self.dev.read_raw()
        rawdata = rawdata.decode('UTF-8')
        volt_data = rawdata[11:-2] #removes header and ending of data
        volt_data = np.array([float(volt_data) for volt_data in volt_data.split(',')])

        t = float(self.dev.query(':WAVeform:XINCrement?'))
        time_data = np.arange(0,t*len(volt_data),t)

        return time_data[0:len(volt_data)], volt_data
    
    @robust
    def scale_offset(self, ch, scale, offset):
        """ 
        Description: The scale_offset function sets the scale and offset parameters of an 
        oscilliscope channel to the scale of choice 

        Input: ch : channel : integer = {1,2,3,4}
               scale : usually in the order of your signal/8 : float
               offset : usually a multiple of the scale : float
               
        Output: None : none : None
        Example: 
        >> acq1.scale_offset(ch, (max_volt-min_volt)/8, -(max_volt-min_volt)/4)

        """
        self.dev.write(':CHANnel{}:SCAL {}'.format(ch,np.round(scale,4)))
        self.dev.write(':CHANnel{}:OFFS {}'.format(ch,np.round(offset,4)))
        
    @robust
    def time_scale_offset(self, scale, offset):
        """ 
        Description: The scale_offset function sets the timing scale and offset parameters of an 
        oscilliscope 

        Input: ch : channel : integer = {1,2,3,4}
               scale : usually in the order of your signal width/4 : float
               offset : usually a multiple of the scale : float
               
        Output: None : none : None
        Example: 
        >> acq1.scale_offset(ch, 1e-6/4, 0)

        """
        self.dev.write('TIMebase:MAIN:SCAL {}'.format(scale))
        self.dev.write('TIMebase:MAIN:OFFS {}'.format(offset))
        
    @robust
    def channel_state(self, ch, state):
        self.dev.write(':CHANnel{}:DISPlay {}'.format(ch,state))
        
    @robust
    def trigger_set(self, ch, edge, level):
        """ 
        Description: Set trigger settings and level

        Input: ch : channel : int = {1,2,3,4}
               edge : edge to trigger on : int = {'POS'=1, 'NEG'=2, 'RFAL'=3}
               level : level to trigger at : float 
        Output: None : None 

        Example: 
        >>trigger_set()

        """
        edges = ['POS','NEG','RFAL']
        self.dev.write(':TRIGger:EDGE:SOURce CHANnel{}'.format(ch))
        self.dev.write(':TRIGger:EDGE:SLOPe {}'.format(edges[edge]))
        self.dev.write(':TRIGger:EDGE:LEVel {}'.format(level))
    
#===========================================================================
# Polarimeter - Thorlabs PAX1000
#=========================================================================== 

class thorlabs_polarimeter:
    def __init__(self,address):
        """ 
        Description: The initialization of the polarimeter class 
        will initialize a singular device using its VISA USB address

        Input: VISA address : string
        Output: device object with all its methods : class object
        Example: 
        >>thorlabs_polarimeter('USB::0x1234::125::A22-5::INSTR')

        thorlabs_polarimeter.class.object...
        """
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.dev.write('SENS:POW:RANG:AUTO 1')
        self.dev.write('SENS:CALC:MODE 9')
        
        
    @robust
    def set_wavelength(self,wavelength):
        """ 
        Description: sets wavelength

        Input: wavelength : float
        Output: None : None
        Example: 
        >>set_wavelength(795)

        """
        if bool(re.search('PAX1000IR1', self.dev.query('*IDN?'))):
            if wavelength < 1080 and wavelength > 600:
                self.dev.write('SENS:CORR:WAV {}'.format(wavelength*1e-9))
            else:
                raise ValueError('Wavelength not supported (600-1080nm)')
        elif bool(re.search('PAX1000IR2', self.dev.query('*IDN?'))):
            if wavelength < 1700 and wavelength > 900:
                self.dev.write('SENS:CORR:WAV {}'.format(wavelength*1e-9))
            else:
                raise ValueError('Wavelength not supported (900-1700nm)')
        else:
            print('check polarimeter connection')

    @robust
    def trigger_data_collection(self):
        """ 
        Description: opens data collection window

        Input: None : None
        Output: None : None
        Example: 
        >>trigger_data_collection()

        """
        self.dev.write('INP:ROT:STAT 1')
        
    @robust
    def close_data_collection(self):
        """ 
        Description: closes data collection window

        Input: None : None
        Output: None : None
        Example: 
        >>close_data_collection()

        """
        self.dev.write('INP:ROT:STAT 0')
    
    @robust
    def get_polarization_params(self):
        """ 
        Description: Returns the current measured Stokes parameters
        for the polarimeter

        Input: None : None
        Output: Stokes Parameters : normalized s1, normalized s2, normalized s3, dop, power : tuple
        Example: 
        >>get_polarzation_params()

        1, 0, 0, 1, 1
        """
        data = self.dev.query('SENS:DATA:LAT?')
        data = data.split(',')
        psi = float(data[9])
        chi = float(data[10])
        dop= float(data[11])
        power = float(data[12])
        s1 = np.cos(2 * psi) * np.cos(2 * chi)
        s2 = np.sin(2 * psi) * np.cos(2 * chi)
        s3 = np.sin(2 * chi)
        return s1, s2, s3, dop, power

    @robust
    def get_raw_data(self):
        """ 
        Description: Returns the raw data read from the polarimeter 

        Input: None : None
        Output: Raw Data from Polarimeter : string
        """
        data = self.dev.query('SENS:DATA:LAT?')
        return data
    
    @robust
    def reset(self):
        """ 
        Description: resets connection with polarimeter

        Input: None : None
        Output: Reset of connection : class method
        """
        self.dev.close()
        time.sleep(1)
        self.dev = self.rm.open_resource(self.address)
        self.dev.write('*RST')
    
    @robust
    def set_power_range(self, ind):
        self.dev.write('SENS:POW:RANG:IND {}'.format(ind))
        
    @robust
    def set_mode(self, mode):
        self.dev.write('SENS:CALC:MODE {}'.format(mode))
        
    @robust
    def set_auto_power_range(self, state : bool):
        self.dev.write('SENS:POW:RANG:AUTO {}'.format(state))
        
        
#================================================================
# Single Quantum SNSPD
#================================================================ 
        
def synchronized_method(method, *args, **kws):
    outer_lock = threading.Lock()
    lock_name = "__" + method.__name__ + "_lock" + "__"

    def sync_method(self, *args, **kws):
        with outer_lock:
            if not hasattr(self, lock_name):
                setattr(self, lock_name, threading.Lock())
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kws)
    sync_method.__name__ = method.__name__
    sync_method.__doc__ = method.__doc__
    sync_method.__module__ = method.__module__
    return sync_method


def _synchronized_method(method):
    return decorate(method, _synchronized_method)


def synchronized_with_attr(lock_name):
    def decorator(method):
        def synced_method(self, *args, **kws):
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kws)
        synced_method.__name__ = method.__name__
        synced_method.__doc__ = method.__doc__
        synced_method.__module__ = method.__module__
        return synced_method
    return decorator

class SQTalk(threading.Thread):
    def __init__(self, TCP_IP_ADR='localhost', TCP_IP_PORT=12000, error_callback=None, TIME_OUT=0.1):
        threading.Thread.__init__(self)
        self.TCP_IP_ADR = TCP_IP_ADR
        self.TCP_IP_PORT = TCP_IP_PORT

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(TIME_OUT)
        self.socket.connect((self.TCP_IP_ADR, self.TCP_IP_PORT))
        self.BUFFER = 10000000
        self.shutdown = False
        self.labelProps = dict()

        self.error_callback = error_callback

        self.lock = threading.Lock()

    @synchronized_method
    def close(self):
        # Print("Closing Socket")
        self.socket.close()
        self.shutdown = True

    @synchronized_method
    def send(self, msg):
        if sys.version_info.major == 3:
            self.socket.send(bytes(msg, "utf-8"))
        if sys.version_info.major == 2:
            self.socket.send(msg)

    def sub_jsons(self, msg):
        """Return sub json strings.
        {}{} will be returned as [{},{}]
        """
        i = 0
        result = []
        split_msg = msg.split('}{')
        for s in range(len(split_msg)):
            if i == 0 and len(split_msg) == 1:
                result.append(split_msg[s])
            elif i == 0 and len(split_msg) > 1:
                result.append(split_msg[s] + "}")
            elif i == len(split_msg) - 1 and len(split_msg) > 1:
                result.append("{" + split_msg[s])
            else:
                result.append("{" + split_msg[s] + "}")
            i += 1
        return result

    @synchronized_method
    def add_labelProps(self, data):
        if "label" in data.keys():
            # After get labelProps, queries also bounds, units etc...
            if isinstance(data["value"], (dict)):
                self.labelProps[data["label"]] = data["value"]
            # General label communication, for example from broadcasts
            else:
                try:
                    self.labelProps[data["label"]
                                    ]["value"] = data["value"]
                except Exception:
                    None

    @synchronized_method
    def check_error(self, data):
        if "label" in data.keys():
            if "Error" in data["label"]:
                self.error_callback(data["value"])

    @synchronized_method
    def get_label(self, label):
        timeout = 10
        dt = .1
        i = 0
        while True:
            if i * dt > timeout:
                raise IOError("Could not acquire label")
            try:
                return self.labelProps[label]
            except Exception:
                time.sleep(dt)
            i += 1

    @synchronized_method
    def get_all_labels(self, label):
        return self.labelProps

    def run(self):
        self.send(json.dumps({"request": "labelProps", "value": "None"}))
        rcv_msg = []

        self.send(json.dumps(
            {"request": "labelProps", "value": "None"}))
        rcv_msg = []

        while self.shutdown is False:
            try:
                rcv = ""+rcv_msg[1]
            except:
                rcv = ""
            data = {}
            r = ""
            while ("\x17" not in rcv) and (self.shutdown == False):
                try:
                    if sys.version_info.major == 3:
                        r = str(self.socket.recv(self.BUFFER), 'utf-8')
                    elif sys.version_info.major == 2:
                        r = self.socket.recv(self.BUFFER)
                except Exception as e:
                    None
                rcv = rcv + r

            rcv_msg = rcv.split("\x17")

            for rcv_line in rcv_msg:
                rcv_split = self.sub_jsons(rcv_line)
                for msg in rcv_split:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        None

                    with self.lock:
                        self.add_labelProps(data)
                        self.check_error(data)


class SQCounts(threading.Thread):
    def __init__(self, TCP_IP_ADR='localhost', TCP_IP_PORT=12345, CNTS_BUFFER=100, TIME_OUT=10):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.rlock = threading.RLock()
        self.TCP_IP_ADR = TCP_IP_ADR
        self.TCP_IP_PORT = TCP_IP_PORT

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(TIME_OUT)
        self.socket.connect((self.TCP_IP_ADR, self.TCP_IP_PORT))
        # self.socket.settimeout(.1)
        self.BUFFER = 1000000
        self.shutdown = False

        self.cnts = []
        self.CNTS_BUFFER = CNTS_BUFFER
        self.n = 0

    @synchronized_method
    def close(self):
        #print("Closing Socket")
        self.socket.close()
        self.shutdown = True

    @synchronized_method
    def get_n(self, n):
        n0 = self.n
        while self.n < n0 + n:
            time.sleep(0.001)
        cnts = self.cnts
        return cnts[-n:]

    def run(self):
        data = []
        while self.shutdown == False:
            if sys.version_info.major == 3:
                try:
                    data_raw = str(self.socket.recv(self.BUFFER), 'utf-8')
                except:
                    data_raw = ""
                    None  # Happens while closing
            elif sys.version_info.major == 2:
                data_raw = self.socket.recv(self.BUFFER)

            data_newline = data_raw.split('\n')

            v = []
            for d in data_newline[0].split(','):
                try:
                    v.append(float(d))
                except:
                    None


            with self.lock:
                self.cnts.append(v)
                # Keep Size of self.cnts
                l = len(self.cnts)
                if l > self.CNTS_BUFFER:
                    self.cnts = self.cnts[l-self.CNTS_BUFFER:]
                self.n += 1

class sq_snspd:
    def __init__(self,address, timeout=10):
        """
        Description: Initialize SNSPD
        
        Input: address : IP-address : string
               timeout : timeout time : float
        Output: None
        """
        
        self.CONTROL_PORT = 12000
        self.COUNTS_PORT = 12345
        self.NUMBER_OF_DETECTORS = 0
        self.address = address
        self.talk = SQTalk(TCP_IP_ADR=self.address,  TCP_IP_PORT=self.CONTROL_PORT,
                           error_callback=self.error, TIME_OUT=timeout)
        
        # Daemonic Thread close when main progam is closed
        self.talk.daemon = True
        self.talk.start()

        self.cnts = SQCounts(TCP_IP_ADR=self.address,
                             TCP_IP_PORT=self.COUNTS_PORT, TIME_OUT=timeout)
        # Daemonic Thread close when main progam is closed
        self.cnts.daemon = True
        self.cnts.start()

        self.NUMBER_OF_DETECTORS = self.talk.get_label("NumberOfDetectors")["value"]
        
        
    def auto_bias_calibration(self, state=True, DarkCounts=[100, 100, 100, 100]):
        """        
        Description: Starts an automatic bias current search. The bias current will be set to match the dark countsself.
        For this function to work properly the detectors should not be exposed to light.
        This function is blocking.
        
        Returns the found bias currents.
        
        Input: None
        Output: bias currents : list
        """     
        msg = json.dumps(dict(command="DarkCountsAutoIV", value=DarkCounts))
        self.talk.send(msg)
        msg = json.dumps(dict(command="AutoCaliBiasCurrents", value=state))
        self.talk.send(msg)
        time.sleep(1)
        while self.talk.get_label("StartAutoIV")["value"] == True:
            time.sleep(.1)        
        return self.talk.get_label("BiasCurrentAutoIV")["value"]
    
    def get_bias_current(self):
        return self.talk.get_label("BiasCurrent")["value"]
    
    def get_number_of_detectors(self):
        return self.talk.get_label("NumberOfDetectors")["value"]
    
    def get_bias_voltages(self):
        """
        Description: get bias voltages
        
        Input: None
        Output: bias voltages : list
        """
        msg = json.dumps(dict(request="BiasVoltage"))
        self.talk.send(msg)
        return self.talk.get_label("BiasVoltage")["value"]

    
    def get_measurement_period(self):
        """
        Description: get measurement period
        
        Input: None
        Output: measurement period : int
        """
        return str(self.talk.get_label("InptMeasurementPeriod")["value"]) + ' ms'
    
    
    def get_number_of_detectors(self):
        """
        Description: get number of detectors 
        
        Input: 
        Output: bias voltages : list
        """
        return self.talk.get_label("NumberOfDetectors")["value"]
    
    
    def get_trigger_level(self):
        """
        Description: get trigger levels in mV
        
        Input: None
        Output: trigger levels : list
        """
        return self.talk.get_label("TriggerLevel")["value"]
    
    def set_bias_current(self,currents : list):
        """
        Description: set bias currents in μA
        
        Input: currents : current in μA : list
        Output: None
        """
        array = currents
        msg = json.dumps(dict(command="SetAllBiasCurrents",
                              label="BiasCurrent", value=array))
        self.talk.send(msg)
        
        
    def set_trigger_level(self,trigger_levels : list):
        """
        Description: set trigger levels in mV
        
        Input: trigger_levels : trigger levels in mV : list
        Output: None
        """
        array = trig
        msg = json.dumps(dict(command="SetAllTriggerLevels",
                              label="TriggerLevel", value=array))
        self.talk.send(msg)
        
        
    def set_measurement_period(self, period : float):
        """
        Description: set measurement period in ms
        
        Input: period : measurement period : float
        Output: None
        """
        msg = json.dumps(
            dict(
                command="SetMeasurementPeriod",
                label="InptMeasurementPeriod",
                value=period))
        
        self.talk.send(msg)
    
        
    def enable_detectors(self, state=True):
        """
        Description: enable detectors
        
        Input: None
        Output: None
        """
        msg = json.dumps(dict(command="DetectorEnable", value=state))
        self.talk.send(msg)
    
    def acquire(self,Ncounts):
        """
        Description: acquire N data points
        
        Input: Ncounts : data points : list
        Output: None
        """
        return self.cnts.get_n(Ncounts)
    
    def close(self):
        """
        Description: close connection
        
        Input: None
        Output: None
        """
        self.talk.close()
        self.talk.join()
        
    def error(self, error_msg):
        """Called in case of an error"""
        print("ERROR DETECTED")
        print(error_msg)
        
#================================================================
# QuTools GMBH QuTAG
#================================================================        

class qutag:
    # ----------------------------------------------------
    # lifetime histogram structure
    class TDC_LftFunction(ctypes.Structure):
        """ Data structure of lifetime function """
        _fields_= [
            ('capacity',ctypes.c_int32),
            ('size',ctypes.c_int32),
            ('binWidth',ctypes.c_int32),
            ('values',ctypes.c_double)]
    
    # hbt histogram structure
    class TDC_HbtFunction(ctypes.Structure):
        """ Data structure of HBT / correlation function """
        _fields_= [
            ('capacity',ctypes.c_int32),
            ('size',ctypes.c_int32),
            ('binWidth',ctypes.c_int32),
            ('indexOffset',ctypes.c_int32),
            ('values',ctypes.c_double)]
            
    def __init__(self):
        """Initializing the quTAG \n\n
        Checking the bit version of Python to load the corresponding DLL \n
        Loading 32 or 64 bit DLL: make sure the wrapper finds the matching DLL in the same folder  \n
        Declary API  \n
        Connect the device by the function self.Initialize()  \n
        Set some parameters
        """
        
        if platform != 'win32':
            raise ValueError('this device is not compatible with your operating system')
        
        package_path = pathlib.Path(pathlib.Path(sys.path[-2]).parents[4])
        dep_path = pathlib.Path(os.path.join('QNCP','dlls'))
        file_path = str(package_path / dep_path.relative_to(dep_path.anchor))
        # check Python bit version
        if sys.maxsize > 2**32:
            # load DLL 64 Bit -------------------------------------------
            dll_name = 'tdcbase_64bit.dll'
            full_path = file_path + os.path.sep + os.path.join("tdcbase_64bit") 
            #print("Python 64 Bit - loading 64 Bit DLL")
        else:
            # load DLL 32 Bit -------------------------------------------
            dll_name = 'tdcbase_32bit.dll'
            full_path = file_path + os.path.sep + os.path.join("DLL_32bit") 
            #print("Python 32 Bit - loading 32 Bit DLL")
        
        # add DLL folder to environment PATH
        os.environ['PATH'] += ';'
        os.environ['PATH'] += full_path

        # load DLL -------------------------------------------
        self.qutools_dll = ctypes.windll.LoadLibrary( full_path )
        
        self.declareAPI()
        self.dev_nr=-1
        
        # wrapper function Initiallize to connect to quTAG
        self.Initialize()
        
        self._bufferSize = 1000000
        self.setBufferSize(self._bufferSize)
        
        self._deviceType = self.getDeviceType()
        self._timebase = self.getTimebase()

        self._StartStopBinCount = 100000
        
        self._featureHBT = self.checkFeatureHBT()
        self._featureLifetime = self.checkFeatureLifetime()
        
        self._HBTBufferSize = 256
        self._LFTBufferSize = 256
        
        #print("Found "+self.devtype_dict[self._deviceType]+" device.")
        
        # Get infos about device
        devType = self._deviceType
        if (devType == self.DEVTYPE_QUTAG):
                print("Found "+self.devtype_dict[self._deviceType]+" device.")
        else:
                print("No suitable device found - demo mode activated")
        
        print("Initialized with self DLL v%f"%(self.getVersion()))

    def declareAPI(self):
        """Declare the API of the DLL with its functions and dictionaries. Should not be executed from the user."""
        # ------- tdcbase.h --------------------------------------------------------
        self.TDC_QUTAG_CHANNELS = 5
        self.TDC_COINC_CHANNELS = 31
        self.TDC_MAX_CHANNEL_NO = 20
        
        # Device types ---------------------------------------
        self.devtype_dict = { 0: 'DEVTYPE_QUTAG', 
            1: 'DEVTYPE_NONE'}
        self.DEVTYPE_QUTAG = 0 # quTAG
        self.DEVTYPE_NONE = 1  # simulated device
        
        # (Output) Fileformats ----------------------------------------
        self.fileformat_dict = { 0: 'ASCII', # ASCII format
            1: 'BINARY', # uncompressed binary format (40B header, 10B/time tag)
            2: 'COMPRESSED', # compressed binary format (40B header, 5B/time tag)
            3: 'RAW', # uncompressed binary without header (for compatibility)
            4: 'NONE' }
            
        self.FILEFORMAT_ASCII = 0
        self.FILEFORMAT_BINARY = 1
        self.FILEFORMAT_COMPRESSED=2
        self.FILEFORMAT_RAW = 3
        self.FILEFORMAT_NONE = 4
        # Signal conditioning --------------------------------
        self.signalcond_dict = { 1: 'LVTTL', # for LVTTL signals: Trigger at 2V rising edge
            2: 'NIM', # for NIM signals: Trigger at -0.6V falling edge
            3: 'MISC', # other signal type: conditioning on, everything optional
            4: 'NONE'}
        
        self.SCOND_LVTTL = 1
        self.SCOND_NIM = 2
        self.SCOND_MISC = 3
        self.SCOND_NONE = 4
        # Type of generated timestamps ----------------------------
        self.simtype_dict = { 0: 'FLAT', # time diffs and channels numbers uniformly ditributed
            1: 'NORMAL', # time diffs normally distributed, channels uniformly
            2: 'NONE'}
        
        self.SIMTYPE_FLAT = 0
        self.SIMTYPE_NORMAL = 1
        self.SIMTYPE_NONE = 2
        
        # Error types (tdcdecl.h) ----------------------------------------
        self.err_dict = {-1: 'unspecified error',
            0 : 'OK, no error', 
            1 : 'Receive timed out', 
            2 : 'No connection was established',
            3 : 'Error accessing the USB driver',
            4 : 'Unknown Error',
            5 : 'Unknown Error',
            6 : 'Unknown Error',
            7 : 'Can''t connect device because already in use',
            8 : 'Unknown error',
            9 : 'Invalid device number used in call',
            10 : 'Parameter in fct. call is out of range',
            11 : 'Failed to open specified file',
            12 : 'Library has not been initialized',
            13 : 'Requested Feature is not enabled',
            14 : 'Requested Feature is not available'}
        
        # function definitions
        self.qutools_dll.TDC_getVersion.argtypes = None
        self.qutools_dll.TDC_getVersion.restype = ctypes.c_double
        self.qutools_dll.TDC_perror.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_perror.restype = ctypes.POINTER(ctypes.c_char_p)
        self.qutools_dll.TDC_getTimebase.argtypes =[ctypes.POINTER(ctypes.c_double)]
        self.qutools_dll.TDC_getTimebase.restype = ctypes.c_int32
        self.qutools_dll.TDC_init.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_init.restype = ctypes.c_int32
        self.qutools_dll.TDC_deInit.argtypes = None
        self.qutools_dll.TDC_deInit.restype = ctypes.c_int32
        self.qutools_dll.TDC_getDevType.argtypes = None
        self.qutools_dll.TDC_getDevType.restype = ctypes.c_int32
        self.qutools_dll.TDC_checkFeatureHbt.argtypes = None
        self.qutools_dll.TDC_checkFeatureHbt.restype = ctypes.c_int32
        #self.qutools_dll.TDC_checkFeatureLifeTime.argtypes = None
        #self.qutools_dll.TDC_checkFeatureLifeTime.restype = ctypes.c_int32
        self.qutools_dll.TDC_getFiveChannelMode.argtypes = None
        self.qutools_dll.TDC_getFiveChannelMode.restype = ctypes.c_int32
        self.qutools_dll.TDC_setFiveChannelMode.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_setFiveChannelMode.restype = ctypes.c_int32
        self.qutools_dll.TDC_getFiveChannelMode.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getFiveChannelMode.restype = ctypes.c_int32
        self.qutools_dll.TDC_preselectSingleStop.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_preselectSingleStop.restype = ctypes.c_int32
        self.qutools_dll.TDC_getSingleStopPreselection.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getSingleStopPreselection.restype = ctypes.c_int32
        self.qutools_dll.TDC_enableChannels.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_enableChannels.restype = ctypes.c_int32
        self.qutools_dll.TDC_getChannelsEnabled.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getChannelsEnabled.restype = ctypes.c_int32
        self.qutools_dll.TDC_enableMarkers.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_enableMarkers.restype = ctypes.c_int32
        self.qutools_dll.TDC_configureSignalConditioning.argtypes = [ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,ctypes.c_double]
        self.qutools_dll.TDC_configureSignalConditioning.restype = ctypes.c_int32
        self.qutools_dll.TDC_getSignalConditioning.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_double)]
        self.qutools_dll.TDC_getSignalConditioning.restype = ctypes.c_int32
        self.qutools_dll.TDC_configureSyncDivider.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.qutools_dll.TDC_configureSyncDivider.restype = ctypes.c_int32
        self.qutools_dll.TDC_getSyncDivider.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getSyncDivider.restype = ctypes.c_int32
        self.qutools_dll.TDC_setCoincidenceWindow.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_setCoincidenceWindow.restype = ctypes.c_int32
        self.qutools_dll.TDC_setExposureTime.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_setExposureTime.restype = ctypes.c_int32
        self.qutools_dll.TDC_getDeviceParams.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getDeviceParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_setChannelDelays.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_setChannelDelays.restype = ctypes.c_int32
        self.qutools_dll.TDC_getChannelDelays.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getChannelDelays.restype = ctypes.c_int32
        self.qutools_dll.TDC_setDeadTime.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.qutools_dll.TDC_setDeadTime.restype = ctypes.c_int32
        self.qutools_dll.TDC_getDeadTime.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getDeadTime.restype = ctypes.c_int32
        self.qutools_dll.TDC_configureSelftest.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.qutools_dll.TDC_configureSelftest.restype = ctypes.c_int32
        self.qutools_dll.TDC_getDataLost.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getDataLost.restype = ctypes.c_int32
        self.qutools_dll.TDC_setTimestampBufferSize.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_setTimestampBufferSize.restype = ctypes.c_int32
        self.qutools_dll.TDC_getTimestampBufferSize.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getTimestampBufferSize.restype = ctypes.c_int32
        self.qutools_dll.TDC_enableTdcInput.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_enableTdcInput.restype = ctypes.c_int32
        self.qutools_dll.TDC_freezeBuffers.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_freezeBuffers.restype = ctypes.c_int32
        self.qutools_dll.TDC_getCoincCounters.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getCoincCounters.restype = ctypes.c_int32
        self.qutools_dll.TDC_getLastTimestamps.argtypes = [ctypes.c_int32,ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int8),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getLastTimestamps.restype = ctypes.c_int32
        self.qutools_dll.TDC_writeTimestamps.argtypes = [ctypes.c_char_p,ctypes.c_int32]
        self.qutools_dll.TDC_writeTimestamps.restype = ctypes.c_int32
        self.qutools_dll.TDC_inputTimestamps.argtypes = [ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int8),ctypes.c_int32]
        self.qutools_dll.TDC_inputTimestamps.restype = ctypes.c_int32
        self.qutools_dll.TDC_readTimestamps.argtypes = [ctypes.c_char_p,ctypes.c_int32]
        self.qutools_dll.TDC_readTimestamps.restype = ctypes.c_int32
        self.qutools_dll.TDC_generateTimestamps.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
        self.qutools_dll.TDC_generateTimestamps.restype = ctypes.c_int32
        
        # ------- tdcmultidev.h ------------------------------------------------------
        self.qutools_dll.TDC_discover.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.qutools_dll.TDC_discover.restype = ctypes.c_int32
        self.qutools_dll.TDC_getDeviceInfo.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_char_p),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getDeviceInfo.restype = ctypes.c_int32
        self.qutools_dll.TDC_connect.argtypes = [ctypes.c_uint32]
        self.qutools_dll.TDC_connect.restype = ctypes.c_int32
        self.qutools_dll.TDC_disconnect.argtypes = [ctypes.c_uint32]
        self.qutools_dll.TDC_disconnect.restype = ctypes.c_int32
        self.qutools_dll.TDC_addressDevice.argtypes = [ctypes.c_uint32]
        self.qutools_dll.TDC_addressDevice.restype = ctypes.c_int32
        self.qutools_dll.TDC_getCurrentAddress.argtypes = [ctypes.c_uint32]
        self.qutools_dll.TDC_getCurrentAddress.restype = ctypes.c_int32
        
        # ------- tdcstartstop.h -----------------------------------------------------
        self.qutools_dll.TDC_enableStartStop.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_enableStartStop.restype = ctypes.c_int32
        self.qutools_dll.TDC_addHistogram.argtypes = [ctypes.c_int32,ctypes.c_int32,ctypes.c_int32]
        self.qutools_dll.TDC_addHistogram.restype = ctypes.c_int32
        self.qutools_dll.TDC_setHistogramParams.argtypes = [ctypes.c_int32,ctypes.c_int32]
        self.qutools_dll.TDC_setHistogramParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHistogramParams.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.c_int32]
        self.qutools_dll.TDC_getHistogramParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_clearAllHistograms.argtypes = None
        self.qutools_dll.TDC_clearAllHistograms.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHistogram.argtypes = [ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int64)]
        self.qutools_dll.TDC_getHistogram.restype = ctypes.c_int32
        
        # ------- tdchbt.h -----------------------------------------------------------
        # type of a HBT model function
        self.fcttype_dict = { 0: 'NONE', 
            1: 'COHERENT', 
            2: 'THERMAL', 
            3: 'SINGLE', 
            4: 'ANTIBUNCH',
            5: 'THERM_JIT',
            6: 'SINGLE_JIT',
            7: 'ANTIB_JIT',
            8: 'THERMAL_OFS',
            9: 'SINGLE_OFS',
            10: 'ANTIB_OFS',
            11: 'THERM_JIT_OFS',
            12: 'SINGLE_JIT_OFS',
            13: 'ANTIB_JIT_OFS'
            }
        
        self.FCTTYPE_NONE = 0
        self.FCTTYPE_COHERENT = 1
        self.FCTTYPE_THERMAL = 2
        self.FCTTYPE_SINGLE = 3
        self.FCTTYPE_ANTIBUNCH = 4
        self.FCTTYPE_THERM_JIT = 5
        self.FCTTYPE_SINGLE_JIT = 6
        self.FCTTYPE_ANTIB_JIT = 7
        self.FCTTYPE_THERMAL_OFS = 8
        self.FCTTYPE_SINGLE_OFS = 9
        self.FCTTYPE_ANTIB_OFS = 10
        self.FCTTYPE_THERM_JIT_OFS = 11
        self.FCTTYPE_SINGLE_JIT_OFS = 12
        self.FCTTYPE_ANTIB_JIT_OFS = 13
        # ----------------------------------------------------
        # function definitions 
        self.qutools_dll.TDC_enableHbt.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_enableHbt.restype = ctypes.c_int32
        self.qutools_dll.TDC_setHbtParams.argtypes = [ctypes.c_int32,ctypes.c_int32]
        self.qutools_dll.TDC_setHbtParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHbtParams.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getHbtParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_setHbtDetectorParams.argtypes = [ctypes.c_double]
        self.qutools_dll.TDC_setHbtDetectorParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHbtDetectorParams.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.qutools_dll.TDC_getHbtDetectorParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_setHbtInput.argtypes = [ctypes.c_int32,ctypes.c_int32]
        self.qutools_dll.TDC_setHbtInput.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHbtInput.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getHbtInput.restype = ctypes.c_int32
        self.qutools_dll.TDC_resetHbtCorrelations.argtypes = None
        self.qutools_dll.TDC_resetHbtCorrelations.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHbtEventCount.argtypes = [ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_double)]
        self.qutools_dll.TDC_getHbtEventCount.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHbtIntegrationTime.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.qutools_dll.TDC_getHbtIntegrationTime.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHbtCorrelations.argtypes = [ctypes.c_int32, ctypes.POINTER(self.TDC_HbtFunction)]
        self.qutools_dll.TDC_getHbtCorrelations.restype = ctypes.c_int32
        self.qutools_dll.TDC_calcHbtG2.argtypes = [ctypes.POINTER(self.TDC_HbtFunction)]
        self.qutools_dll.TDC_calcHbtG2.restype = ctypes.c_int32
        self.qutools_dll.TDC_fitHbtG2.argtypes = [ctypes.POINTER(self.TDC_HbtFunction),ctypes.c_int32,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_fitHbtG2.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHbtFitStartParams.argtypes = [ctypes.c_int32,ctypes.POINTER(ctypes.c_double)]
        self.qutools_dll.TDC_getHbtFitStartParams.restype = ctypes.POINTER(ctypes.c_double)
        self.qutools_dll.TDC_calcHbtModelFct.argtypes = [ctypes.c_int32,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(self.TDC_HbtFunction)]
        self.qutools_dll.TDC_calcHbtModelFct.restype = ctypes.c_int32
        self.qutools_dll.TDC_generateHbtDemo.argtypes = [ctypes.c_int32,ctypes.POINTER(ctypes.c_double),ctypes.c_double]
        self.qutools_dll.TDC_generateHbtDemo.restype = ctypes.c_int32
        self.qutools_dll.TDC_createHbtFunction.argtypes = None
        self.qutools_dll.TDC_createHbtFunction.restype = ctypes.POINTER(self.TDC_HbtFunction)
        self.qutools_dll.TDC_releaseHbtFunction.argtypes = [ctypes.POINTER(self.TDC_HbtFunction)]
        self.qutools_dll.TDC_releaseHbtFunction.restype = None
        self.qutools_dll.TDC_analyseHbtFunction.argtypes = [ctypes.POINTER(self.TDC_HbtFunction),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_double),ctypes.c_int32]
        self.qutools_dll.TDC_analyseHbtFunction.restype = ctypes.c_int32
        
        # ------- tdclifetm.h --------------------------------------------------------
        self.LFT_PARAM_SIZE = 4
        # type of a lifetime model function
        self.lfttype_dict = {0: 'NONE',
            1: 'EXP',
            2: 'DBL_EXP',
            3: 'KOHLRAUSCH'
            }
        self.LFTTYPE_NONE = 0
        self.LFTTYPE_EXP = 1
        self.LFTTYPE_DBL_EXP = 2
        self.LFTTYPE_KOHLRAUSCH = 3

        # function definitions
        self.qutools_dll.TDC_enableLft.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_enableLft.restype = ctypes.c_int32
        self.qutools_dll.TDC_setLftStartInput.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_setLftStartInput.restype = ctypes.c_int32
        self.qutools_dll.TDC_addLftHistogram.argtypes = [ctypes.c_int32,ctypes.c_int32]
        self.qutools_dll.TDC_addLftHistogram.restype = ctypes.c_int32
        self.qutools_dll.TDC_getLftStartInput.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getLftStartInput.restype = ctypes.c_int32
        self.qutools_dll.TDC_setLftParams.argtypes = [ctypes.c_int32,ctypes.c_int32]
        self.qutools_dll.TDC_setLftParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_getLftParams.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getLftParams.restype = ctypes.c_int32
        self.qutools_dll.TDC_resetLftHistograms.argtypes = None
        self.qutools_dll.TDC_resetLftHistograms.restype = ctypes.c_int32
        self.qutools_dll.TDC_createLftFunction.argtypes = None
        self.qutools_dll.TDC_createLftFunction.restype = ctypes.POINTER(self.TDC_LftFunction)
        self.qutools_dll.TDC_releaseLftFunction.argtypes = [ctypes.POINTER(self.TDC_LftFunction)]
        self.qutools_dll.TDC_releaseLftFunction.restype = None
        self.qutools_dll.TDC_analyseLftFunction.argtypes = [ctypes.POINTER(self.TDC_LftFunction),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_double),ctypes.c_int32]
        self.qutools_dll.TDC_analyseLftFunction.restype = None
        self.qutools_dll.TDC_getLftHistogram.argtypes = [ctypes.c_int32,ctypes.c_int32,ctypes.POINTER(self.TDC_LftFunction),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int64)]
        self.qutools_dll.TDC_getLftHistogram.restype = ctypes.c_int32
        self.qutools_dll.TDC_calcLftModelFct.argtypes = [ctypes.c_int32,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(self.TDC_LftFunction)]
        self.qutools_dll.TDC_calcLftModelFct.restype = ctypes.c_int32
        self.qutools_dll.TDC_generateLftDemo.argtypes = [ctypes.c_int32,ctypes.POINTER(ctypes.c_double),ctypes.c_double]
        self.qutools_dll.TDC_generateLftDemo.restype = ctypes.c_int32
        self.qutools_dll.TDC_fitLftHistogram.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.c_int32,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_fitLftHistogram.restype = ctypes.c_int32
        
        
        # ------- tdchg2.h --------------------------------------------------------
        self.qutools_dll.TDC_enableHg2.restype = ctypes.c_int32
        self.qutools_dll.TDC_enableHg2.argtypes = [ctypes.c_int32]
        self.qutools_dll.TDC_getHg2Input.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHg2Input.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getHg2Params.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHg2Params.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_getHg2Raw.restype = ctypes.c_int32
        self.qutools_dll.TDC_getHg2Raw.argtypes = [ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)]
        self.qutools_dll.TDC_resetHg2Correlations.restype = ctypes.c_int32
        self.qutools_dll.TDC_resetHg2Correlations.argtypes = None
        self.qutools_dll.TDC_setHg2Input.restype = ctypes.c_int32
        self.qutools_dll.TDC_setHg2Input.argtypes = [ctypes.c_int32,ctypes.c_int32,ctypes.c_int32]
        self.qutools_dll.TDC_setHg2Params.restype = ctypes.c_int32
        self.qutools_dll.TDC_setHg2Params.argtypes = [ctypes.c_int32,ctypes.c_int32]
        self.qutools_dll.TDC_calcHg2G2.restype = ctypes.c_int32
        self.qutools_dll.TDC_calcHg2G2.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int32),ctypes.c_int32]
        self.qutools_dll.TDC_calcHg2Tcp.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.c_int32]
        self.qutools_dll.TDC_calcHg2Tcp.restype = ctypes.c_int32
        
# Init --------------------------------------------------------------    
    def Initialize(self):
        """Initializing the quTAG by DLL cunction TDC_init with device number -1 \n

        @rtype: string
        @return: Returns error code via dictionary
    
        """
        ans = self.qutools_dll.TDC_init(self.dev_nr)
        
        if (ans != 0):
            print("Error in TDC_init: " + self.err_dict[ans])
        return ans

    def deInitialize(self):
        """Deinitializing the quTAG by DLL cunction TDC_deInit \n
         Important to clear the connection to reconnect to the device \n
         Return error code via dictionary
        """
        ans = self.qutools_dll.TDC_deInit()
        
        if (ans != 0): # from the documentation: "never fails"
            print("Error in TDC_deInit: " + self.err_dict[ans])
        return ans

# Device Info -------------------------------------------------------------
    def getVersion(self):
        return self.qutools_dll.TDC_getVersion()
    
    def getTimebase(self):
        timebase = ctypes.c_double()
        ans = self.qutools_dll.TDC_getTimebase(ctypes.byref(timebase))
        if (ans!=0):
            print("Error in TDC_getTimebase: "+self.err_dict[ans])
        return timebase.value
        
    def getDeviceType(self):
        ans = self.qutools_dll.TDC_getDevType()
        return ans
        
    def checkFeatureHBT(self):
        ans = self.qutools_dll.TDC_checkFeatureHbt()
        return ans == 1
		
    def checkFeatureLifetime(self):
        ans = self.qutools_dll.TDC_checkFeatureLifeTime()
        return ans == 1
            
    def checkFeatureFiveChan(self):
        ans = self.qutools_dll.TDC_checkFeatureFiveChan()
        return ans == 1
		
    def getFiveChannelMode(self):
        enable = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getFiveChannelMode(ctypes.byref(enable))
        if (ans != 0):
            print("Error in TDC_getFiveChannelMode: "+self.err_dict[ans])
        return enable.value==1
    
    def getSingleStopPreselection(self):
        enable = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getSingleStopPreselection(ctypes.byref(enable))
        if (ans != 0):
            print("Error in TDC_getSingleStopPreselection: "+self.err_dict[ans])
        return enable.value==1
    
    def preselectSingleStop(self, boolsch):
        """
        The input parameter of the Python function is a bool and gets changed to integer 0 or 1 for TDC_preselectSingleStop \n
        @type single: bool
        @param single: True and False for enable or disable

        @rtype: string
        @return: Returns the error code via the dictionary
        """
        if (boolsch):
            enable = 1
        else:
            enable = 0
        ans = self.qutools_dll.TDC_preselectSingleStop(enable)
        if (ans != 0):
            print("Error in TDC_preselectSingleStop: "+self.err_dict[ans])
        return self.err_dict[ans]
# multiple devices ---------------------------------    
    def addressDevice(self,deviceNumber):
        ans = self.qutools_dll.TDC_addressDevice(deviceNumber)
        if (ans!=0):
            print("Error in TDC_addressDevice: "+self.err_dict[ans])
        return ans
    
    def connect(self,deviceNumber):
        ans = self.qutools_dll.TDC_connect(deviceNumber)
        if (ans!=0):
            print("Error in TDC_connect: "+self.err_dict[ans])
        return ans
    
    def disconnect(self,deviceNumber):
        ans = self.qutools_dll.TDC_disconnect(deviceNumber)
        if (ans!=0):
            print("Error in TDC_disconnect: "+self.err_dict[ans])
        return ans
    
    def discover(self):
        devCount = ctypes.c_uint32()
        ans = self.qutools_dll.TDC_discover(ctypes.byref(devCount))
        if (ans!=0):
            print("Error in TDC_discover: "+self.err_dict[ans])
        return devCount.value
    
    def getCurrentAddress(self):
        devNo = ctype.c_unit32()
        ans = self.qutools_dll.TDC_getCurrentAddress(ctypes.byref(devNo))
        if (ans!=0):
            print("Error in TDC_getCurrentAddress: "+self.err_dict[ans])
        return devNo.value
        
    def getDeviceInfo(self,deviceNumber):
        devicetype = ctypes.c_int32()
        deviceid = ctypes.c_int32()
        serialnumnber=ctypes.c_char_p()
        connected = ctypes.s_int32()
        
        ans = self.qutools_dll.TDC_getDeviceInfo(deviceNumber,ctypes.byref(devicetype), ctypes.byref(deviceid), ctypes.byref(serialnumber), ctypes.byref(connected))
        
        if (ans!=0):
            print("Error in TDC_getDeviceInfo: "+self.err_dict[ans])
            
        return (devicetype.value, deviceid.value, serialnumber.value,connected.value)
        
# Configure Channels ----------------------------------------------------------------
    def getSignalConditioning(self, channel):
        edg = ctypes.c_int32()
        threshold = ctypes.c_double()
        
        ans = self.qutools_dll.TDC_getSignalConditioning(channel, ctypes.byref(edg), ctypes.byref(threshold))
        
        if (ans != 0):
            print("Error in TDC_getSignalConditioning: "+self.err_dict[ans])
            
        return (edg.value == 1, threshold.value)
    
    def setSignalConditioning(self, channel, conditioning, edge, threshold):
        if edge:
            edge_value = 1 # True: Rising
        else:
            edge_value = 0 # False: Falling
        
        ans = self.qutools_dll.TDC_configureSignalConditioning(channel,conditioning,edge_value,threshold)
        if (ans != 0):
            print("Error in TDC_configureSignalConditioning: "+self.err_dict[ans])
        return ans
    
    def getDivider(self):
        divider = ctypes.c_int32()
        reconstruct = ctypes.c_bool()
        ans = self.qutools_dll.TDC_getSyncDivider(ctypes.byref(divider), ctypes.byref(reconstruct))    
        
        if (ans != 0):
            print("Error in TDC_getSyncDivider: " + self.err_dict[ans])
            
        return (divider.value, reconstruct.value)
        
    def setDivider(self, divider, reconstruct):
        # allowed values:
        # - quTAG: 1, 2, 4, 8
        ans = self.qutools_dll.TDC_configureSyncDivider(divider, reconstruct)    
        if (ans != 0):
            print("Error in TDC_configureSyncDivider: " + self.err_dict[ans])
        return ans
        
    def getChannelDelays(self):
        delays = np.zeros(int(8), dtype=np.int32)
        ans = self.qutools_dll.TDC_getChannelDelays(delays.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if (ans != 0):
            print("Error in TDC_getChannelDelays: " + self.err_dict[ans])
        return delays
        
    def setChannelDelays(self, delays):
        ans = self.qutools_dll.TDC_setChannelDelays(delays.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if (ans != 0):
            print("Error in TDC_setChannelDelays: " + self.err_dict[ans])
        return ans
        
    def getDeadTime(self,chn):        
        #chn = ctypes.c_int32()
        deadTime = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getDeadTime(chn, ctypes.byref(deadTime))
        if (ans != 0):
            print("Error in TDC_getDeadTime: " + self.err_dict[ans])
        return deadTime.value
        
    def setDeadTime(self,chn,deadTime):
        ans = self.qutools_dll.TDC_setDeadTime(chn,deadTime)
        if (ans != 0):
            print("Error in TDC_setDeadTime: " + self.err_dict[ans])
        return ans
        
    def setFiveChannelMode(self,enable):
        if enable:
            ena = 1
        else:
            ena = 0
        ans = self.qutools_dll.TDC_setFiveChannelMode(ena)
        if (ans != 0):
            print("Error in TDC_setFiveChannelMode: "+self.err_dict[ans])
        return ans
    
    def enableTDCInput(self, enable):
        if enable:
            value = 1 # enable input
        else:
            value = 0 # disable input
        
        ans = self.qutools_dll.TDC_enableTdcInput(value)
        if (ans != 0):
            print("Error in TDC_enableTdcInput: "+self.err_dict[ans])
            
        return ans
    
    def enableChannels(self, channels):
        if channels:
            bitstring = ''
            for k in range(max(channels)+1):
                if k in channels:
                    bitstring = '1' + bitstring
                else:
                    bitstring = '0' + bitstring
        else:
            bitstring = '0'

        channelMask = int(bitstring, 2)
        ans = self.qutools_dll.TDC_enableChannels(channelMask)
        if ans != 0:
            print("Error in TDC_enableChannels: "+self.err_dict[ans])
        
        return ans
    
    def getChannelsEnabled(self):
        channelMask = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getChannelsEnabled(ctypes.byref(channelMask))
        
        channels = [0 for i in range(self.TDC_QUTAG_CHANNELS)]
        mask = channelMask.value
        i=1
        while mask > 0:
            channels[self.TDC_QUTAG_CHANNELS-i] = mask.value % 2
            mask //= 2
            i += 1
            if i > self.TDC_QUTAG_CHANNELS:
                print("Error in computing channelMask (getChannelsEnabled).")
                break
                
        if ans != 0:
            print("Error in TDC_enableChannels: "+self.err_dict[ans])
        return channels

    def enableMarkers(self, markers):
        if markers:
            bitstring = ''
            for k in range(max(markers)+1):
                if k in markers:
                    bitstring = '1' + bitstring
                else:
                    bitstring = '0' + bitstring
        else:
            bitstring = '0'

        markerMask = int(bitstring, 2)
        ans = self.qutools_dll.TDC_enableMarkers(markerMask)
        if ans != 0:
            print("Error in TDC_enableMarkers: "+self.err_dict[ans])
        
        return ans
        
# Define Measurements -------------------------------------------------------
    def setCoincidenceWindow(self, coincWin):
        ans = self.qutools_dll.TDC_setCoincidenceWindow(coincWin)
        if ans != 0:
            print("Error in TDC_setCoincidenceWindows: "+self.err_dict[ans])
        return 0
        
    def setExposureTime(self, expTime):
        ans = self.qutools_dll.TDC_setExposureTime(expTime)
        if ans != 0:
            print("Error in TDC_setExposureTime: "+self.err_dict[ans])
        return ans
        
    def getDeviceParams(self):
        chn = ctypes.c_int32()
        coinc = ctypes.c_int32()
        exptime = ctypes.c_int32()
        
        ans = self.qutools_dll.TDC_getDeviceParams( ctypes.byref(coinc), ctypes.byref(exptime))
        if ans != 0:
            print("Error in TDC_getDeviceParams: "+self.err_dict[ans])
        return (chn.value, coinc.value, exptime.value)

# Self test ---------------------------------------------------------------------
    def configureSelftest(self, channels, period, burstSize, burstDist):
        if channels:
            bitstring = ''
            for k in range(max(channels)+1):
                if k in channels:
                    bitstring = '1' + bitstring
                else:
                    bitstring = '0' + bitstring
        else:
            bitstring = '0'

        channelMask = int(bitstring, 2)
        ans = self.qutools_dll.TDC_configureSelftest(channelMask,period,burstSize,burstDist)
        if ans != 0:
            print("Error in TDC_configureSelftest: "+self.err_dict[ans])
            
        return ans
        
    def generateTimestamps(self, simtype, par, count):
        ans = self.qutools_dll.TDC_generateTimestamps(simtype,ctypes.byref(par),count)
        if ans != 0:
            print("Error in TDC_generateTimestamps: "+self.err_dict[ans])
        return ans
        
# Timestamping ---------------------------------------------------------
    def getBufferSize(self):
        sz = ctype.c_int32()
        ans = self.qutools_dll.TDC_getTimestampBufferSize(ctypes.byref(sz))
        if ans != 0:
            print("Error in TDC_getTimestampBufferSize: "+self.err_dict[ans])
        return sz.value
    
    def setBufferSize(self, size):
        self._bufferSize = size
        ans = self.qutools_dll.TDC_setTimestampBufferSize(size)
        if ans != 0:
            print("Error in TDC_setTimestampBufferSize: "+self.err_dict[ans])
        return ans
        
    def getDataLost(self):
        lost = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getDataLost(ctypes.byref(lost))
        if ans != 0:
            print("Error in TDC_getDataLost: "+self.err_dict[ans])
        return lost.value
        		
		
    def freezeBuffers(self, freeze):
        if freeze:
            freeze_value = 1
        else:
            freeze_value = 0
        ans = self.qutools_dll.TDC_freezeBuffers(freeze_value)
        if ans != 0:
            print("Error in TDC_freezeBuffers: "+self.err_dict[ans])
            
        return ans
    
    def getLastTimestamps(self,reset):
        timestamps = np.zeros(int(self._bufferSize), dtype=np.int64)
        channels = np.zeros(int(self._bufferSize), dtype=np.int8)
        valid = ctypes.c_int32()

        ans = self.qutools_dll.TDC_getLastTimestamps(reset,timestamps.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),channels.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),ctypes.byref(valid))
        if ans != 0: # "never fails"
            print("Error in TDC_getLastTimestamps: "+self.err_dict[ans])
            
        return timestamps,channels,valid.value
    
# File IO -------------------------------------------
    def writeTimestamps(self, filename, fileformat):
        filename = filename.encode('utf-8')
        ans = self.qutools_dll.TDC_writeTimestamps(filename,fileformat)
        if ans != 0:
            print("Error in TDC_writeTimestamps: "+self.err_dict[ans])
        return ans
        
    def inputTimestamps(self, timestamps,channels,count):
        ans = self.qutools_dll.TDC_inputTimestamps(ctypes.byref(timestamps),ctypes.byref(channels),count)
        if ans != 0:
            print("Error in TDC_inputTimestamps: "+self.err_dict[ans])
        return ans
    
    def readTimestamps(self, filename, fileformat):
        filename = filename.encode('utf-8')
        ans = self.qutools_dll.TDC_readTimestamps(filename,fileformat)
        if ans != 0:
            print("Error in TDC_readTimestamps: "+self.err_dict[ans])
        return ans
        
# Counting --------------------------------------------
    def getCoincCounters(self):
        data = np.zeros(int(31),dtype=np.int32)
        update = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getCoincCounters(data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),ctypes.byref(update))
        if ans != 0: # "never fails"
            print("Error in TDC_getCoincCounters: "+self.err_dict[ans])
        return (data,update.value)



# Start-Stop --------------------------------------------------------
    def enableStartStop(self, enable):
        if enable:
            ena_value = 1
        else:
            ena_value = 0
        ans = self.qutools_dll.TDC_enableStartStop(ena_value)
        if ans != 0:
            print("Error in TDC_enableStartStop: "+self.err_dict[ans])
        return ans
    
    def addHistogram(self, startChannel, stopChannel, enable):
        self.enableStartStop(True)
        if enable:
            ena_value = 1
        else:
            ena_value = 0
        ans = self.qutools_dll.TDC_addHistogram(startChannel, stopChannel,ena_value)
        if ans != 0:
            print("Error in TDC_addHistogram: "+self.err_dict[ans])
        return ans
    
    def setHistogramParams(self, binWidth, binCount):
        self._StartStopBinCount = binCount
        ans = self.qutools_dll.TDC_setHistogramParams(binWidth,binCount)
        if ans != 0:
            print("Error in TDC_setHistogramParams: "+self.err_dict[ans])
        return ans
    
    def getHistogramParams(self):
        binWidth = ctypes.c_int32()
        binCount = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getHistogramParams(ctypes.byref(binWidth),ctypes.byref(binCount))
        if ans != 0:
            print("Error in TDC_getHistogramParams: "+self.err_dict[ans])
        return (binWidth.value, binCount.value)
    
    def clearAllHistograms(self):
        ans = self.qutools_dll.TDC_clearAllHistograms()
        if ans != 0:
            print("Error in TDC_clearAllHistograms: "+self.err_dict[ans])
        return ans
        
    def getHistogram(self, chanA, chanB, reset):
        if reset:
            reset_value = 1
        else:
            reset_value = 0
        data = np.zeros(self._StartStopBinCount, dtype=np.int32)
        count = ctypes.c_int32()
        tooSmall = ctypes.c_int32()
        tooLarge = ctypes.c_int32()
        starts = ctypes.c_int32()
        stops = ctypes.c_int32()
        expTime = ctypes.c_int64()
        ans = self.qutools_dll.TDC_getHistogram(chanA,chanB,reset_value,data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),ctypes.byref(count),ctypes.byref(tooSmall),ctypes.byref(tooLarge),ctypes.byref(starts),ctypes.byref(stops),ctypes.byref(expTime))
        if ans != 0:
            print("Error in TDC_getHistogram: "+self.err_dict[ans])
        
        return (data,count.value,tooSmall.value,tooLarge.value,starts.value,stops.value,expTime.value)
        
#   Lifetime ----------------------------------------------------------
    def enableLFT(self,enable):
        if enable:
            ena = 1
        else:
            ena = 0
        ans = self.qutools_dll.TDC_enableLft(ena)
        if ans != 0:
            print("Error in TDC_enableLft: "+self.err_dict[ans])
        return ans
        
    def setLFTParams(self,binWidth,binCount):
        self._LFTBufferSize = binCount
        ans = self.qutools_dll.TDC_setLftParams(binWidth, binCount)
        if ans != 0:
            print("Error in TDC_setLftParams: "+self.err_dict[ans])
        return ans

    def getLFTParams(self):
        binWidth = ctypes.c_int32()
        binCount = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getLftParams(ctypes.byref(binWidth), ctypes.byref(binCount))
        if ans != 0:
            print("Error in TDC_getLftParams: "+self.err_dict[ans])
        return binWidth.value, binCount.value

    def setLFTStartInput(self,startChannel):
        ans = self.qutools_dll.TDC_setLftStartInput(startChannel)
        if ans != 0:
            print("Error in TDC_setLftStartInput: "+self.err_dict[ans])
        return ans

    def getLFTStartInput(self):
        startChannel = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getLFTStartInput(ctype.byref(startChannel))
        if ans != 0:
            print("Error in TDC_getLFTStartInput: "+self.err_dict[ans])
        return startChannel.value

    def resetLFTHistograms(self):
        ans = self.qutools_dll.TDC_resetLftHistograms()
        if ans != 0:
            print("Error in TDC_resetLftHistrograms: "+self.err_dict[ans])
        return ans

    def createLFTFunction(self):
        LFTfunction = self.qutools_dll.TDC_createLftFunction()
        return LFTfunction

    def releaseLFTFunction(self, LFTfunction):
        self.qutools_dll.TDC_releaseLftFunction(LFTfunction)
        return 0

    def addLFTHistogram(self,stopchannel,enable):
        if enable:
            ena = 1
        else:
            ena = 0
        
        ans = self.qutools_dll.TDC_addLftHistogram(stopchannel,ena)
        if ans != 0:
            print("Error in TDC_addLftHistogram: "+self.err_dict[ans])
        return ans
        
    def analyseLFTFunction(self,lft):
        capacity = ctypes.c_int32()
        size = ctypes.c_int32()
        binWidth = ctypes.c_int32()
        values = np.zeros(self._LFTBufferSize, dtype=np.double)
        
        self.qutools_dll.TDC_analyseLftFunction (lft, ctypes.byref(capacity), ctypes.byref(size), ctypes.byref(binWidth), values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._LFTBufferSize)
        return (capacity.value, size.value, binWidth.value, values)

    def getLFTHistogram(self,channel,reset, lft):
        print("getLFTHistogram")
        tooBig = ctypes.c_int32()
        startevt = ctypes.c_int32()
        stopevt = ctypes.c_int32()
        expTime = ctypes.c_int64()
        if reset:
            resetvalue = 1
        else:
            resetvalue = 0
            
        ans = self.qutools_dll.TDC_getLftHistogram(channel, resetvalue, lft, ctypes.byref(tooBig), ctypes.byref(startevt), ctypes.byref(stopevt), ctypes.byref(expTime))
        if ans != 0:
            print("Error in TDC_getLFTHistogram: "+self.err_dict[ans])
        return (tooBig.value, startevt.value, stopevt.value, expTime.value, lft)

    def calcLFTModelFCT(self,lfttype,params,lftfunction):
        c_params = np.zeros(self.LFT_PARAM_SIZE,dtype=np.double)
        for i in range(len(params)):
            if (i < self.LFT_PARAM_SIZE):
                c_params[i] = params[i]
            else:
                break
        ans = self.qutools.TDC_calcLftModelFct(lfttype,c_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),lftfunction)
        if ans != 0:
            print("Error in TDC_calcLftModelFct: "+self.err_dict[ans])
        return ans

    def generateLFTDemo(self,lfttype,params,noiseLv):
        c_params = np.zeros(self.LFT_PARAM_SIZE,dtype=np.double)
        for i in range(len(params)):
            if (i < self.LFT_PARAM_SIZE):
                c_params[i] = params[i]
            else:
                break
        ans = selfg.qutools.TDC_generateLftDemo(lfttype,c_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),noiseLv)
        if ans != 0:
            print("Error in TDC_generateLftDemo: "+self.err_dict[ans])
        return ans

    def fitLFTHistogram(self,lft,lfttype, startParams):
        c_params = np.zeros(self.LFT_PARAM_SIZE,dtype=np.double)
        for i in range(len(startParams)):
            if (i < self.LFT_PARAM_SIZE):
                c_params[i] = startParams[i]
            else:
                break
        fitParams=np.zeros(4,dtype=np.double)
        iterations = ctypes.c_int32()
        
        ans = self.qutools_dll.TDC_fitLftHistogram(lft,lfttype,c_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),fitParams.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.byref(iterations))
        if ans != 0:
            print("Error in TDC_fitLftHistogram: "+self.err_dict[ans])
        return (fitParams,iterations.value)

#   HBT ---------------------------------------------------------------
    def enableHBT(self, enable):
        if enable:
            ena_value = 1
        else:
            ena_value = 0
        ans = self.qutools_dll.TDC_enableHbt(ena_value)
        if ans != 0:
            print("Error in TDC_enableHbt: "+self.err_dict[ans])
        return ans
    
    def setHBTParams(self, binWidth, binCount):
        ans = self.qutools_dll.TDC_setHbtParams(binWidth,binCount)
        self._HBTBufferSize = binCount * 2 - 1
        if ans != 0:
            print("Error in TDC_setHbtParams: "+self.err_dict[ans])
        return ans
    
    def getHBTParams(self):
        binWidth = ctypes.c_int32()
        binCount = ctypes.c_int32()
        ans = self.qutools_dll.TDC_setHbtParams(ctypes.byref(binWidth),ctypes.byref(binCount))
        if ans != 0:
            print("Error in TDC_getHbtParams: "+self.err_dict[ans])
        return (binWidth.value, binCount.value)
    
    def setHBTDetectorParams(self, jitter):
        ans = self.qutools_dll.TDC_setHbtDetectorParams(jitter)
        if ans != 0:
            print("Error in TDC_setHbtDetectorParams: "+self.err_dict[ans])
        return ans
    
    def getHBTDetectorParams(self):
        jitter = ctypes.c_double()
        ans = self.qutools_dll.TDC_getHbtDetectorParams(ctypes.byref(jitter))
        if ans != 0:
            print("Error in TDC_getHbtdetectorParams: "+self.err_dict[ans])
        return jitter.value
    
    
    def setHBTInput(self, channel1, channel2):
        ans = self.qutools_dll.TDC_setHbtInput(channel1, channel2)
        if ans != 0:
            print("Error in TDC_setHbtInput: "+self.err_dict[ans])
        return ans
    
    
    def getHBTInput(self):
        channel1=ctypes.c_int32()
        channel2=ctypes.c_int32()
        ans = self.qutools_dll.TDC_getHbtInput(ctypes.byref(channel1), ctypes.byref(channel2))
        if ans != 0:
            print("Error in TDC_getHbtInput: "+self.err_dict[ans])
        return (channel1.value,channel2.value)
    
    def resetHBTCorrelations(self):
        ans = self.qutools_dll.TDC_resetHbtCorrelations()
        if ans != 0:
            print("Error in TDC_resetHbtCorrelations: "+self.err_dict[ans])
        return ans
    
    def getHBTEventCount(self):
        totalCount = ctypes.c_int64()
        lastCount = ctypes.c_int64()
        lastRate = ctypes.c_double()
        ans = self.qutools_dll.TDC_getHbtEventCount(ctypes.byref(totalCount), ctypes.byref(lastCount), ctypes.byref(lastRate))
        if ans != 0:
            print("Error in TDC_getHbtEventCount: "+self.err_dict[ans])
        return (totalCount.value,lastCount.value,lastRate.value)
    
    def getHBTIntegrationTime(self):
        intTime = ctypes.c_double()
        ans = self.qutools_dll.TDC_getHbtIntegrationTime(ctypes.byref(intTime))
        if ans != 0:
            print("Error in TDC_getHbtIntegrationTime: "+self.err_dict[ans])
        return intTime.value
    
    def getHBTCorrelations(self, forward, hbtfunction):
        ans = self.qutools_dll.TDC_getHbtCorrelations(forward,hbtfunction)
        if ans != 0:
            print("Error in TDC_getHbtCorrelations: "+self.err_dict[ans])
        return ans
    
    def calcHBTG2(self, hbtfunction):
        ans = self.qutools_dll.TDC_calcHbtG2(hbtfunction)
        if ans != 0:
            print("Error in TDC_calcHbtG2: "+self.err_dict[ans])
        return ans
    
    def fitHBTG2(self, hbtfunction, fitType, startParams):
        c_params = np.zeros(self.HBT_PARAM_SIZE,dtype=np.double)
        for i in range(len(startParams)):
            if (i < self.HBT_PARAM_SIZE):
                c_params[i] = startParams[i]
            else:
                break
        fitParams = np.zeros(self.HBT_PARAM_SIZE,dtype=np.double)
        iterations = ctypes.c_int32()
        
        ans = self.qutools_dll.TDC_fitHbtG2(hbtfunction,fitType,c_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),fitParams.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.byref(iterations))
        if ans != 0:
            print("Error in TDC_fitHbtG2: "+self.err_dict[ans])
        return (fitParams,iterations.value)
    
    def getHBTFitStartParams(self, fctType):
        fitParams = np.zeros(self.HBT_PARAM_SIZE,dtype=np.double)
        ans = self.qutools_dll.TDC_getHbtFitStartParams(fctType, fitParams.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if ans != 0:
            print("Error in TDC_getHbtFitStartParams: "+self.err_dict[ans])
        return fitParams
    
    def calcHBTModelFct(self, fctType, params, hbtfunction):
        c_params = np.zeros(self.HBT_PARAM_SIZE,dtype=np.double)
        for i in range(len(params)):
            if (i < self.HBT_PARAM_SIZE):
                c_params[i] = params[i]
            else:
                break
        ans = self.qutools_dll.TDC_calcHbtModelFct(fctType,c_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),hbtfunction)
        if ans != 0:
            print("Error in TDC_calcHbtModelFct: "+self.err_dict[ans])
        return ans
    
    def generateHBTDemo(self, fctType, params, noiseLv):
        c_params = np.zeros(self.HBT_PARAM_SIZE,dtype=np.double)
        for i in range(len(params)):
            if (i < self.HBT_PARAM_SIZE):
                c_params[i] = params[i]
            else:
                break
        ans = self.qutools_dll.TDC_generateHbtDemo(fctType,c_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),noiseLv)
        if ans != 0:
            print("Error in TDC_generateHbtDemo: "+self.err_dict[ans])
        return ans
        
    def createHBTFunction(self):
        return self.qutools_dll.TDC_createHbtFunction()
        
    def releaseHBTFunction(self, hbtfunction):
        self.qutools_dll.TDC_releaseHbtFunction(hbtfunction)
        return 0
    
    def analyzeHBTFunction(self, hbtfunction):
        capacity = ctypes.c_int32()
        size = ctypes.c_int32()
        binWidth = ctypes.c_int32()
        iOffset = ctypes.c_int32()
        values = np.zeros(self._HBTBufferSize,dtype=np.double)
        self.qutools_dll.TDC_analyseHbtFunction(hbtfunction,ctypes.byref(capacity),ctypes.byref(size),ctypes.byref(binWidth),ctypes.byref(iOffset),values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),self._HBTBufferSize)
        
        return (capacity.value,size.value,binWidth.value,iOffset.value,values)

#   Heralded g(2) ----------------------------------------------------------
    def enableHg2(self,enable):
        if enable:
            ena = 1
        else:
            ena = 0
        ans = self.qutools_dll.TDC_enableHg2(ena)
        if ans != 0:
            print("Error in TDC_enableLft: " + self.err_dict[ans])
        return self.err_dict[ans]
        
    def setHg2Params(self, binWidth, binCount):
        ans = self.qutools_dll.TDC_setHg2Params(binWidth,binCount)
        if ans != 0:
            print("Error in TDC_setHg2Params: " + self.err_dict[ans])
        return self.err_dict[ans]

    def getHg2Params(self):
        binWidth = ctypes.c_int32()
        binCount = ctypes.c_int32()
        ans = self.qutools_dll.TDC_getHg2Params(ctypes.byref(binWidth),ctypes.byref(binCount))
        if ans != 0:
            print("Error in TDC_getHg2Params: "+self.err_dict[ans])
        return (binWidth.value, binCount.value)
        
    def setHg2Input(self, idler, channel1, channel2):
        ans = self.qutools_dll.TDC_setHg2Input(idler, channel1, channel2)
        if ans != 0:
            print("Error in TDC_setHg2Input: "+self.err_dict[ans])
        return self.err_dict[ans]

    def getHg2Input(self):
        idler=ctypes.c_int32()
        channel1=ctypes.c_int32()
        channel2=ctypes.c_int32()
        ans = self.qutools_dll.TDC_getHg2Input(ctypes.byref(idler),ctypes.byref(channel1), ctypes.byref(channel2))
        if ans != 0:
            print("Error in TDC_getHg2Input: "+self.err_dict[ans])
        return (idler.value,channel1.value,channel2.value)

    def resetHg2Correlations(self):
        ans = self.qutools_dll.TDC_resetHg2Correlations()
        if ans != 0:
            print("Error in TDC_resetHg2Correlations: "+self.err_dict[ans])
        return self.err_dict[ans]

    def calcHg2G2(self, reset):
        if reset:
            resetvalue = 1
        else:
            resetvalue = 0
        binCount = self.getHg2Params()[1]
        buffer = np.zeros( binCount , dtype=np.double)
        bufSize = ctypes.c_int32( binCount )
        
        ans = self.qutools_dll.TDC_calcHg2G2(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(bufSize), resetvalue)
        if ans != 0:
            print("Error in TDC_calcHg2G2: "+self.err_dict[ans])
        return buffer

   
    def calcHg2Tcp1D(self, reset):
        if reset:
            resetvalue = 1
        else:
            resetvalue = 0
        binCount = self.getHg2Params()[1]
        binCount2 = binCount*binCount
        buffer=np.zeros(binCount2, dtype=np.int64)
        bufSize = ctypes.c_int32(binCount2)
        
        ans = self.qutools_dll.TDC_calcHg2Tcp1D(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.byref(bufSize), resetvalue)
        if ans != 0:
            print("Error in TDC_calcHg2Tcp1D: "+self.err_dict[ans])
        return buffer 

       
    def getHg2Raw(self): 
        evtIdler=ctypes.c_int64()
        evtCoinc=ctypes.c_int64()
        binCount = self.getHg2Params()[1]
        bufSsi= np.zeros(binCount, dtype=np.int64)
        bufS2i = np.zeros(binCount, dtype=np.int64)
        bufSize = ctypes.c_int32(binCount)

        ans = self.qutools_dll.TDC_getHg2Raw(ctypes.byref(evtIdler), ctypes.byref(evtCoinc),bufSsi.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), bufS2i.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.byref(bufSize))
        if ans != 0:
            print("Error in TDC_getHg2Raw: "+self.err_dict[ans])
        return (evtIdler.value, evtCoinc.value, bufSsi, bufS2i)