import pyvisa
from pyvisa.constants import Parity,StopBits
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from scipy.signal import find_peaks
import inspect
import vxi11
import time
import socket
import serial
import select
from struct import unpack
from collections import OrderedDict
import six
CRLF = b'\r\n'

def robust(method, *arg):
    def robust_method(self, *arg):
        try:
            result = method(self,*arg)
        except:
            self.dev.close()
            time.sleep(1)
            self.__init__(self.address)
            result = method(self,*arg)
        return result
        
    robust_method.__name__ = method.__name__
    robust_method.__doc__ = method.__doc__
    robust_method.__module__ = method.__module__
    return robust_method

#================================================================
# Valon_5015('Pete' or 'Ringo')
#================================================================
class Valon_5015:
    def __init__(self,address):  #'TCPIP::<IP ADDRESS>::23::SOCKET'
        self.rm = pyvisa.ResourceManager()  
        self.address = address 
        self.dev = self.rm.open_resource(self.address)
        self.dev.read_termination = '-->'
        self.clear()
    
    @robust
    def off(self): 
        self.write('OEN OFF;')  # turn off RF output
        self.write('PDN OFF;')  # turn off synth output
    
    @robust
    def on(self):
        self.write('OEN ON;')  # turn on RF output
        self.write('PDN ON;');  # turn on synth output

    @robust
    def freq(self,*f):  # MHz, enter or read frequency
        if bool(f) == True:   # assign value (only the first in the tuple)
            if type(f[0]) == str:
                self.write('Freq {};'.format(f[0]));
            else:
                self.write('Freq {} MHz;'.format(f[0]));
        else:   # when no input is entered, read actual frequency 
            self.__readOut = self.query('Freq?;')
            self.__freqAct = float(re.search('(?<=Act )\S+',self.__readOut).group())
            return self.__freqAct

    @robust
    def lev(self,*l):  # MHz
        if bool(l) == True:  # assign level
            if type(l[0]) == str:
                self.__pwr = float(re.search('[0-9]*(\.)*[0-9]*',l[0]).group())
                self.write('PWR {};'.format(self.__pwr));
            else:
                self.write('PWR {};'.format(l[0]));
        else:  # when empty input, read actual level
            self.__readOut = self.query('PWR?;')
            self.__levAct = float(re.search('(?<=PWR ).*(?=\;)',self.__readOut).group())
            return self.__levAct
        
    @robust        
    def write(self,arg):
        self.clear()    
        self.dev.write(arg)
        
    # @robust
    # def read(self):
    #     return self.dev.read()

    @robust
    def query(self, arg):
        self.write(arg)
        result = self.dev.read()
        return result

    @robust
    def clear(self):  # clear the device command. Very important for Valon!
        self.dev.clear()
#     def close(self):
#         self.dev.close()

#================================================================
# Function Generator - Rigol DSG800 series
#================================================================

class Rigol_DSG800:
    
    def __init__(self,address):
        self.address = address 
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
    
    @robust
    def off(self):
        self.dev.write(':OUTput OFF;')  # turn on RF output
    
    @robust    
    def on(self):
        self.dev.write(':OUTput On;')  # turn on RF output
    
    @robust
    def freq(self,*f):  # MHz
        if bool(f) == True:
            __f = f[0]
            if type(__f) == str:
                __f = re.sub(r'\s','',__f)    # delet the space in between, if any
                if re.search('([a-zA-Z])',__f) == None:  # MHz
                    __f = __f + 'MHz'
                self.dev.write(':Freq {};'.format(__f));
            else:
                self.dev.write(':Freq {}MHz;'.format(__f)); # no space in between
        else:
            __readOut = self.dev.query(':Freq?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq
    
    @robust        
    def lev(self,*l):  # MHz
        if bool(l) == True:
            __l = l[0]
            if type(__l) == str:
                __l = re.sub(r'\s','',__l)    # delet the space in between, if any
                self.dev.write(':LEV {};'.format(__l));
            else:
                self.dev.write(':LEV {}dBm;'.format(__l));# no space in between
        else:
            __readOut = self.dev.query(':LEV?')
            __lev = float(re.search('.*(?=\n)',__readOut).group())
            return __lev

#================================================================
# Function Generator - Rigol DG4000 series
#================================================================
class Rigol_DG4000:
    def __init__(self,address):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
    
    @robust
    def reset(self): #Reset
        self.dev.write("*RST")
    
    @robust
    def off(self,*ch):  # default: both
        if bool(ch) == True: # turn single output off
            for channel in ch:
                self.dev.write(':OUTput{} OFF'.format(channel))  
        else:  # turn both off
            self.dev.write(':OUTput1 OFF') 
            self.dev.write(':OUTput2 OFF')
    @robust        
    def on(self,*ch): # default: both
        if bool(ch) == True: # turn single output on
            for channel in ch:
                self.dev.write(':OUTput{} ON'.format(channel))  
        else:  # turn both no
            self.dev.write(':OUTput1 ON') 
            self.dev.write(':OUTput2 ON')

    @staticmethod
    def power_vrms(v,z):
        """
        Calculate power using Vrms
        Input: Vrms, impedance Z
        Output: power (W)
        """
        return v**2/z

    @staticmethod
    def power_vpp(v,z):
        """
        Calculate power using Vpp
        Input: Vpp, impedance Z
        Output: power (W)
        
        """
        return v**2/z/8

    @staticmethod
    def power_dbm(dbm):
        """
        Convert power from dBm to W
        Input: dBm
        Output: power (W)
        
        """
        return pow(10, -3 + dbm/10)

    @robust        
    def __Hz(self,f):  # in Hz, support unit. Default: MHz
        """
        Description: Sets all frequencies to MHz Unit (Tested 07/15/2022)

        Input: f : frequency : float

        Output: f * 1e6 : frequency in MHz : float
        """
        if type(f) == str:
            if re.search('[mM]',f) != None:
                return 1e6*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[kK]',f) != None:
                return 1e3*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[hH]',f) != None:
                return 1*float(re.sub('[a-zA-Z]','',f))
        else: # float, or str that contains only numbers
            return float(f)*1e6
    @robust    
    def output_impedence(self, ch, *load):
        if bool(load) == False:
            load = 'INF'
        else:
            load = load[0]
        self.dev.write(':OUTPut{}:LOAD {}'.format(ch,load))
    
    @robust
    def freq(self,ch,*f):
        if bool(f) == True:   # assign value (only the first in the tuple)
            freq = self.__Hz(f[0])
            try:
                freq_apply = min(freq,  eval('self.hz_max_{}'.format(ch)))
            except:
                freq_apply = freq
            self.dev.write(':SOURCe{}:Freq {}'.format(ch, freq_apply));
            if freq_apply < freq:
                print('Warning: channel {} output frequency is too high. Reduced to {} Hz'.
                      format(ch,eval('self.hz_max_{}'.format(ch))))
        else:   # when no input is entered, read actual frequency 
            __readOut = self.dev.query(':SOURCe{}:Freq?'.format(ch))
            __freq = float(re.search('.*(?=\n)',__readOut).group())
            return float(__freq)*1e-6  # MHz
    
    @robust
    def freq_max(self, ch, *f):
        """
        Setup the frequency limit.
        
        Input: channel, frequency max (any unit)
        
        Output: maximum frequency (Hz)
        """
        if bool(f):
            hz_max = f[0]
            exec("self.hz_max_{} = {} ".format(ch, self.__Hz(hz_max)))
        else:
            try:
                return eval("self.hz_max_{}".format(ch))   # 'Hz'
            except:
                print('Warning: maximum output frequency has not been specified yet.' )
    
    @robust
    def lev(self,ch,*v):
        if bool(v) == True:
            __v = v[0]
            if type(__v) == str:
                __lev = float(re.sub('[a-zA-Z]','',__v))  # unitless value
                # mVPP or mVRMS
                if re.search('(mv)',__v,re.IGNORECASE) != None:
                    __lev = 1e-3 * __lev
                # apply voltage 
                if re.search('(vrms)',__v,re.IGNORECASE) != None:  #  VRMS
                    self.dev.write(':SOURCe{}:VOLTage:UNIT VRMS'.format(ch))
                    try:
                        lev_apply = min(__lev, eval('self.vpp_max_{}/2/pow(2,1/2)'.format(ch)) )
                    except:
                        lev_apply = __lev
                elif re.search('(dbm)',__v,re.IGNORECASE) != None:  # dBm
                    self.dev.write(':SOURCe{}:VOLTage:UNIT DBM'.format(ch))
                    try:
                        lev_apply = min(__lev, eval('10*np.log10(1000*self.power_max_{})'.format(ch)) )
                    except:
                        lev_apply = __lev
                elif re.search('(vpp)',__v,re.IGNORECASE) != None:  # VPP
                    self.dev.write(':SOURCe{}:VOLTage:UNIT VPP'.format(ch))
                    try:
                        lev_apply = min(__lev, eval('self.vpp_max_{}'.format(ch)) )
                    except:
                        lev_apply = __lev
                self.dev.write(':SOURCe{}:VOLTage {}'.format(ch,lev_apply))
            else:  # default: [Vpp] 
                __lev = __v
                try:
                    lev_apply = min(__lev, eval('self.vpp_max_{}'.format(ch)) )
                except:
                    lev_apply = __lev
                self.dev.write(':SOURCe{}:VOLTage:UNIT VPP'.format(ch))
                self.dev.write(':SOURCe{}:VOLTage {}'.format(ch,lev_apply))
            # print out warning
            if lev_apply < __lev:
                print('Protection Warning: output {} amplitude is too high. Reduced to limit value.'.format(ch))
        else:
            __readOut = self.dev.query(':SOURCe{}:VOLTage?'.format(ch))
            __lev = float(re.search('.*(?=\n)',__readOut).group())
            __readOut = self.dev.query(':SOURCe{}:VOLTage:UNIT?'.format(ch))
            __unit = re.search('.*(?=\n)',__readOut).group()
            return __lev, __unit
    
    @robust
    def lev_max(self,ch, *v):
        """
        Set maximum output level, in any units.
        * Note that Vrms - Vpp conversion only holds for sinusoidal wave!
        Input: channel, maximum voltage/power.
        
        Output: maximum amplitude, (and maximum power if imepance is not inf).
        """
        z_string = self.dev.query(':OUTPut{}:LOAD?'.format(ch))
        z = float(re.search('.*(?=\n)',z_string).group())
        # assign max voltage
        if bool(v) == True:  
            __v = v[0]
            if type(__v) == str:  # with unit
                lev = float(re.sub('[a-zA-Z]','',__v))  # unitless value
                if re.search('(vrms)',__v,re.IGNORECASE) != None:  #  VRMS
                    exec("self.vpp_max_{} = lev * 2 * pow(2, 1/2) ".format(ch))
                    if z != np.inf:
                        exec("self.power_max_{} = power_vrms(lev,z)".format(ch))
                elif re.search('(vpp)',__v,re.IGNORECASE) != None:  # Vpp
                    exec("self.vpp_max_{} = lev".format(ch))
                    if z != np.inf:
                        exec("self.power_max_{} = power_vpp(lev,z)".format(ch))
                elif (re.search('(dbm)',__v,re.IGNORECASE) != None):  # dbm
                    exec("self.power_max_{} = power_dbm(lev)".format(ch))   # W
                    if  (z != np.inf):
                        exec("self.vpp_max_{} = 2*pow(2,1/2)*pow(self.power_max_{} * z, 1/2)"
                             .format(ch,ch))
                # mVPP or mVRMS
                if re.search('(mv)',__v,re.IGNORECASE) != None:
                    exec("self.vpp_max_{} = 1e-3 * self.vpp_max_{}".format(ch,ch))
                    exec("self.power_max_{} = 1e-6 * self.power_max_{}".format(ch,ch))
            else:  # default: [Vpp] 
                exec("self.vpp_max_{} = v[0]".format(ch))
                exec("self.power_max_{} = power_vpp(v[0],z)".format(ch))
        else:  # read self.power_max
            try:
                return eval("self.vpp_max_{}".format(ch))   # 'Vpp'
            except:
                try:
                    if bool(eval("self.power_max_{}".format(ch))) == True:
                        print('Warning: maximum output voltage cannot be specified because of infinite impedance.' )
                except:
                    print('Warning: maximum output level has not been specified yet.' )
    @robust    
    def offset(self,ch,offset):  # V_DC
        self.dev.write(':SOURCe{}:VOLTage:OFFSet {}'.format(ch,offset))

    @robust    
    def phase(self,ch,phase):
        self.dev.write(':SOURCe{}:PHASe {}'.format(ch,phase))
        
    @staticmethod    
    def gaussian(t,mu,FWHM,a): 
        sigma = (FWHM)/(2*np.sqrt(2*np.log(2)))
        amplitude = np.sqrt(2*np.pi*sigma**2)*a
        return amplitude*( 1/(sigma * np.sqrt(2*np.pi) ) )*np.exp( -((t-mu)**2 / (2*sigma**2)) )
    
    @staticmethod
    def square(t,leadingedge,width,amp): #square pulse with duty cycle
        return np.piecewise(t,[(t<=leadingedge),((t>leadingedge) & (t<leadingedge+width)),(t>=leadingedge+width)],[0,amp,0])
    
    @staticmethod
    def normalize(waveform):
        """
        Description: Normalizes data for arbitrary waveform design, points are limited (Tested 04/03/2022)
        to -1 to 1 Volt

        Input: data : waveform : np.array or list

        Output: np.array(waveform)/np.absolute(max(waveform)) : normalized data :  np.array or list
        """
        factor = max([np.abs(max(waveform)),np.abs(min(waveform))])
        return np.array(waveform)/np.absolute(factor)
        
    @robust
    def arbitrary(self, ch, signal_width, waveform, *arg):
        """
        Description: Allows one to set and create arbitrary waveform output (Tested 05/27/2022) 

        Input: ch : channel : int
               signal_width : width (time) of the argument in seconds : float
               waveform : method or list of values : method or array
               arg* : arguments of the waveform function : misc

        Output: None : class method
        """
        buffer_size = 2**14
        if inspect.ismethod(waveform) == True or inspect.isfunction(waveform) == True:
            t = np.linspace(0,signal_width,buffer_size)
            data = waveform(t,*arg)
            midpoint = (max(data)-min(data))/2
            centered_data = data - midpoint
            datastring = ",".join(map(str,np.round(self.normalize(centered_data),4)))
        else:
            data = waveform
            midpoint = (max(data)-min(data))/2
            centered_data = data - midpoint
            datastring = ",".join(map(str,np.round(self.normalize(centered_data),4)))
    
        self.dev.write('SOURCE{}:Freq {}'.format(ch, 1/signal_width))
        self.dev.write("SOURCE{}:TRACE:DATA VOLATILE,".format(ch) + datastring)
        self.dev.write("SOURCE{}:FUNC VOLATILE,".format(ch))
        self.dev.write("SOURCE{}:VOLTAGE:LOW {}".format(ch,min(centered_data)))
        self.dev.write("SOURCE{}:VOLTAGE:HIGH {}".format(ch,max(centered_data)))
        self.dev.write('SOURCe{}:VOLTage:OFFSet {}'.format(ch,midpoint))
        self.dev.write("SOURCE{}:PHASE:SYNC".format(ch))
    
    @robust
    def burst_mode(self, ch, mode, cycles):
        """
        Description: Allows on to use burst functionallity (Tested 04/03/2022)

        Input: ch : channel : int
               mode : burst mode : int = {'TRIG' = 0, 'GAT' = 1,'INF' = 2}
               cycles : cycles : int

        Output: None : class method
        """
        modes = ['TRIGgered','GATed','INFinity']
        self.dev.write('SOURce{}:BURSt:MODE {}'.format(ch,modes[mode]))
        self.dev.write('SOURce{}:BURSt:TRIGger:SOURce EXT'.format(ch))
        self.dev.write('SOURce{}:BURSt:NCYCles {}'.format(ch,cycles))
        self.dev.write('SOURce{}:BURSt ON'.format(ch))
        
    @robust
    def burst_state(self, ch, state):
        states = ['OFF','ON']
        self.dev.write('SOURce{}:BURSt {}'.format(ch, states[state]))
        
    @robust    
    def DC(self, ch, offset):
        """
        Description: Enables DC Mode (Tested 04/03/2022)

        Input: ch : channel : int
               offset : DC Offset : float

        Output: None : class method
        """
        self.dev.write("SOURCE{}:FUNC DC".format(ch))
        self.offset(ch,offset)
    
    @robust
    def ext_trig(self,ch):
        self.dev.write("SOURCE{}:BURST:TRIG:SOUR EXT".format(ch))
        
    @robust
    def arbitrary_burst(self,ch,signal_width,cycles,waveform,*arg):
        """
        Description: Uses N cycle burst functionallity of the Rigol DG4000 Series. Best 
        suited for external trigger.

        Input: ch : channel : int
               signal_width : width (time) of the argument in seconds : float
               cycles : cycles : int
               func : method or list of values : method or array
               arg* : arguments of the function : misc
        Output: None : class method
        """
        if inspect.ismethod(waveform) == True or inspect.isfunction(waveform) == True:
            t = np.linspace(0,signal_width,1000)
            data = waveform(t,*arg)
            datastring = ",".join(map(str, self.normalize(data)))
        else:
            data = waveform
            datastring = ",".join(map(str, self.normalize(data)))

        self.dev.write('SOURCE{}:Freq {}'.format(ch, 1/signal_width))
        self.dev.write("OUTPUT{} ON".format(ch))
        self.dev.write("SOURCE{}:TRACE:DATA VOLATILE,".format(ch)+ datastring)
        self.dev.write("SOURCE{}:VOLTAGE:UNIT VPP".format(ch))
        self.dev.write("SOURCE{}:VOLTAGE:AMPL {}".format(ch,2*max(data)))
        self.dev.write("SOURCE{}:VOLTAGE:OFFSET 0".format(ch))
        self.dev.write("SOURCE{}:PHASE 0".format(ch))
        self.dev.write("SOURCE{}:PHASE:SYNC".format(ch))

        #triggered burst
        self.dev.write("SOURCE{}:BURST ON".format(ch))
        self.dev.write("SOURCE{}:BURST:NCYC {}".format(ch,cycles))
        self.dev.write("SOURCE{}:BURST:MODE:TRIG".format(ch))
        self.ext_trig(ch)
        
        
#===========================================================================
#MOGLabs
#===========================================================================
"""
moglabs device class
Simplifies communication with moglabs devices

Compatible with both python2 and python3

v1.2: Fixed Unicode ambiguities, added explicit close(), fixed USB error in recv_raw()
v1.1: Made compatible with both python2 and python3
v1.0: Initial release

(c) MOGLabs 2016--2021
http://www.moglabs.com/
"""

# Handles communication with devices
class MOGDevice(object):
    def __init__(self,addr,port=None,timeout=1,check=True):
        assert len(addr), 'No address specified'
        self.dev = None                        
        
        # is it a COM port?
        if addr.startswith('COM') or addr == 'USB':
            if port is not None: addr = 'COM%d'%port
            addr = addr.split(' ',1)[0]
            self.connection = addr
            self.is_usb = True
        else:
            if not ':' in addr:
                if port is None: port=7802
                addr = '%s:%d'%(addr,port)
            self.connection = addr
            self.is_usb = False
        self.reconnect(timeout,check)

    def __repr__(self):
        """Returns a simple string representation of the connection"""
        return 'MOGDevice("%s")'%self.connection
    
    def close(self):
        """Close any active connection. Can be reconnected at a later time"""
        if self.connected():
            self.dev.close()
            self.dev = None
    
    def reconnect(self,timeout=1,check=True):
        """Reestablish connection with unit"""
        # close the handle if open - this is _required_ on USB
        self.close()
        if self.is_usb:
            try:
                self.dev = serial.Serial(self.connection, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=timeout, writeTimeout=0)
            except serial.SerialException as E:
                raise RuntimeError(E.args[0].split(':',1)[0])
        else:
            self.dev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dev.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.dev.settimeout(timeout)
            addr, port = self.connection.split(':')
            self.dev.connect((addr,int(port)))
        # check the connection?
        if check:
            try:
                self.info = self.ask('info')
            except Exception as E:
                raise RuntimeError('Device did not respond to query')
    
    def connected(self):
        """Returns True if a connection has been established, but does not validate the channel is still open"""
        return self.dev is not None
        
    def _check(self):
        """Assers that the device is connected"""
        assert self.connected(), 'Not connected'
                 
    def versions(self):
        """Returns a dictionary of device version information"""
        verstr = self.ask('version')
        if verstr == 'Command not defined':
            raise RuntimeError('Incompatible firmware')
        # does the version string define components?
        vers = {}
        if ':' in verstr:
            # old versions are LF-separated, new are comma-separated
            tk = ',' if ',' in verstr else '\n'
            for l in verstr.split(tk):
                if l.startswith('OK'): continue
                n,v = l.split(':',2)
                v = v.strip()
                if ' ' in v: v = v.rsplit(' ',2)[1].strip()
                vers[n.strip()] = v
        else:
            # just the micro
            vers['UC'] = verstr.strip()
        return vers

    def cmd(self,cmd):
        """Send the specified command, and check the response is OK. Returns response in Unicode"""
        resp = self.ask(cmd)
        if resp.startswith('OK'):
            return resp
        else:
            raise RuntimeError(resp)
        
    def ask(self,cmd):
        """Send followed by receive, returning response in Unicode"""
        # check if there's any response waiting on the line
        self.flush()
        self.send(cmd)
        resp = self.recv().strip()
        if resp.startswith('ERR:'):
            raise RuntimeError(resp[4:].strip())
        return resp
        
    def ask_dict(self,cmd):
        """Send a request which returns a dictionary response, with keys and values in Unicode"""
        resp = self.ask(cmd)
        # might start with "OK"
        if resp.startswith('OK'): resp = resp[3:].strip()
        # expect a colon in there
        if not ':' in resp: raise RuntimeError('Response to '+repr(cmd)+' not a dictionary')
        # response could be comma-delimited (new) or newline-delimited (old)
        splitchar = ',' if ',' in resp else '\n'
        # construct the dict (but retain the original key order)
        vals = OrderedDict()
        for entry in resp.split(splitchar):
            key, val = entry.split(':')
            vals[key.strip()] = val.strip()
        return vals
        
    def ask_bin(self,cmd):
        """Send a request which returns a binary response, returned in Bytes"""
        self.send(cmd)
        head = self.recv_raw(4)
        # is it an error message?
        if head == b'ERR:': raise RuntimeError(self.recv().strip())
        datalen = unpack('<L',head)[0]
        data = self.recv_raw(datalen)
        if len(data) != datalen: raise RuntimeError('Binary response block has incorrect length')
        return data
    
    def send(self,cmd):
        """Send command, appending newline if not present"""
        if hasattr(cmd,'encode'):  cmd = cmd.encode()
        if not cmd.endswith(CRLF): cmd += CRLF
        self.send_raw(cmd)
    
    def has_data(self,timeout=0):
        """Returns True if there is data waiting on the line, otherwise False"""
        self._check()
        if self.is_usb:
            try:
                if self.dev.inWaiting(): return True
                if timeout == 0: return False
                time.sleep(timeout)
                return self.dev.inWaiting() > 0
            except serial.SerialException: # will raise an exception if the device is not connected
                return False
        else:
            sel = select.select([self.dev],[],[],timeout)
            return len(sel[0])>0
        
    def flush(self,timeout=0,buffer=256):
        self._check()       
        dat = ''
        while self.has_data(timeout):
            chunk = self.recv(buffer)
            # handle the case where we get binary rubbish and prevent TypeError
            if isinstance(chunk,six.binary_type) and not isinstance(dat,six.binary_type): dat = dat.encode()
            dat += chunk
        return dat
    
    def recv(self,buffer=256):
        """Receive a line of data from the device, returned as Unicode"""
        self._check()
        if self.is_usb:
            data = self.dev.readline(buffer)
            if len(data):
                while self.has_data(timeout=0):
                    segment = self.dev.readline(buffer)
                    if len(segment) == 0: break
                    data += segment
            if len(data) == 0: raise RuntimeError('Timed out')
        else:
            data = b''
            while True:
                data += self.dev.recv(buffer)
                timeout = 0 if data.endswith(CRLF) else 0.1
                if not self.has_data(timeout): break
        try:
            # try to return the result as a Unicode string
            return data.decode()
        except UnicodeDecodeError:
            # even though we EXPECTED a string, we got raw data so return it as bytes
            return data
    
    def send_raw(self,cmd):
        """Send, without appending newline"""
        self._check()
        if self.is_usb:
            return self.dev.write(cmd)
        else:
            return self.dev.send(cmd)
    
    def recv_raw(self,size):
        """Receive exactly 'size' bytes"""
        self._check()
        parts = []
        tout = time.time() + self.get_timeout()
        while size > 0:
            if self.is_usb:
                chunk = self.dev.read(min(size,0x2000))
            else:
                chunk = self.dev.recv(min(size,0x2000))
            if time.time() > tout:
                raise DeviceError('timed out')
            parts.append(chunk)
            size -= len(chunk)
        buf = b''.join(parts)
        return buf
        
    def get_timeout(self):
        """Return the connection timeout, in seconds"""
        self._check()
        if self.is_usb:
            return self.dev.timeout
        else:
            return self.dev.gettimeout()
            
    def set_timeout(self,val = None):
        """Change the timeout to the specified value, in seconds"""
        self._check()
        old = self.get_timeout()
        if val is not None:
            if self.is_usb:
                self.dev.timeout = val
            else:
                self.dev.settimeout(val)
            return old

        
def load_script(filename):
    """Loads a script of commands for line-by-line execution, removing comments"""
    with open(filename,"rU") as f:  # open in universal mode
        for linenum, line in enumerate(f):
            # remove comments
            line = line.split('#',1)[0]
            # trim spaces
            line = line.strip()
            if len(line) == 0: continue
            # for debugging purposes it's helpful to know which line of the file is being executed
            yield linenum+1, line

#-----------------------------------------------------------------------
# Driver for MOGLabs
class MOGLabs:
    def __init__(self,address): #<IP ADDRESS>
        self.address = address
        self.dev = MOGDevice(self.address)

    def off(self,*ch):
        if bool(ch) == True:
            for channel in ch:
                self.dev.cmd('Off,{}'.format(channel))
        else:
            self.dev.cmd('Off,1')
    def on(self,*ch):
        if bool(ch) == True:
            for channel in ch:
                self.dev.cmd('On,{}'.format(channel))
        else:
            self.dev.cmd('On,1')

    def freq(self,ch,*frequency):
        if bool(frequency):
            if type(frequency) == str:
                self.dev.cmd('FREQ,{},{}'.format(ch,frequency[0]))
            else:
                self.dev.cmd('FREQ,{},{} MHz'.format(ch,frequency[0]))
        else: # read out
            f = self.dev.ask('FREQ,{}'.format(ch))
            f = float(re.search('[0-9.]*',f).group() )
            return f

    def lev(self,ch,*amplitude):
        if bool(amplitude):
            if type(amplitude) == str:
                self.dev.cmd('POW,{},{}'.format(ch,amplitude[0]))
            else:
                self.dev.cmd('POW,{},{} dBm'.format(ch,amplitude[0]))
        else: # read out
            l = self.dev.ask('POW,{}'.format(ch))
            return l            
            

    def am(self,ch=1):  # turn on amplitude modulation
        self.on(ch)    # turn on RF output
        self.dev.cmd('POW,{},0 dBm'.format(ch) )
        self.dev.cmd('LIM,{},30 dBm'.format(ch) )
        self.dev.cmd('MDN,{},AMPL,ON'.format(ch) )
        self.dev.cmd('GAIN,{},AMPL,6815'.format(ch) )
        self.dev.cmd('MDN,{},FREQ,OFF'.format(ch) )

    def norm(self,ch=1):
        self.dev.cmd('MDN,{},AMPL,OFF'.format(ch) )          # AM modulation off
        self.dev.cmd('POW,{},30 dBm'.format(ch) )

    def trig_ext(self, ch=1):
        self.dev.cmd('on,{}'.format(ch)) # Has to turn on first. ask 'OK: CH1 output changed'
        self.dev.cmd('extio,mode,{},off,toggle'.format(ch))  # mode has to be set first
        self.dev.cmd('extio,enable,{},OFF'.format(ch)) # has to be done after toggle mode is set

        
#===========================================================================
# Quantum Composers
#===========================================================================

def sleep_method(method, *arg):
    t_sleep = 50e-3
    def sleeping_method(self, *arg):
        method(self,*arg)
        time.sleep(t_sleep)
        
    sleeping_method.__name__ = method.__name__
    sleeping_method.__doc__ = method.__doc__
    sleeping_method.__module__ = method.__module__
    return sleeping_method

class Quantum_Composers:
    
    @sleep_method
    def __init__(self,address,*arg):
        if arg:
            self.arg = arg
            self.address = address
            self.baud_rate = self.arg[0]
            self.rm = pyvisa.ResourceManager()
            self.dev = self.rm.open_resource(self.address, # same as 'ASRL5::INSTR'
                                  baud_rate = self.baud_rate, # must identify
                                  data_bits = 8,
                                  parity = Parity.none,
                                  stop_bits = StopBits.one)
            self.mux_reset()   # clear all multiplexer
            self.lev()    # set all outputs to TTL
            self.t_sleep = 50e-3
            self.digit   = 11   # important! round evertying to 11 digits
        else:
            self.arg = False
            self.address = address
            self.rm = pyvisa.ResourceManager()
            self.dev = self.rm.open_resource(self.address)
            self.mux_reset()   # clear all multiplexer
            self.lev()    # set all outputs to TTL
            self.dev.clear()
            self.t_sleep = 50e-3
            self.digit   = 11   # important! round evertying to 11 digits

    def rd(self,x,*digit):
        if bool(digit) == False:
            return round(x,self.digit)
        else:
            return round(x,digit)
        
    @sleep_method
    @robust
    def t0(self,t):  # clock T0
        t = self.rd(t)
        self.dev.write(':PULSE0:PER {}'.format(t))
    
    @sleep_method
    @robust
    def norm(self,*ch):   # normal mode, no wait
        if bool(ch) == True:  # specified channel
            for __ch in ch:
                if __ch > 0: # channel model
                    self.dev.write(':Pulse{}:CMODe NORMal'.format(__ch))
                elif __ch == 0:  # T0 mode
                    self.dev.write(':Pulse0:MODe NORMal')
        else:
            for __ch in range(1,9):  # all channels and T0
                self.dev.write(':Pulse{}:CMODe NORMal'.format(__ch))
                self.dev.write(':Pulse0:MODe NORMal')
                
    @sleep_method   
    @robust
    def wid(self,ch,w):
        w = self.rd(w)
        self.dev.write(':PULSE{}:WIDth {}'.format(ch,w))
        
    @sleep_method
    @robust
    def dly(self,ch,d):
        d = self.rd(d)
        self.dev.write(':PULSE{}:DELay {}'.format(ch,d))
        
    @sleep_method
    @robust
    def pol(self,ch,p):
        self.dev.write(':PULSE{}:POL {}'.format(ch,p))
        
    @sleep_method
    @robust
    def wcount(self,ch,w):  # wait number of T0 before enable output       
        self.dev.write(':PULSE{}:WCOunter {}'.format(ch,w))
        
    @sleep_method
    @robust
    def dcycl(self,ch,on,off):   # channel duty cycle
        self.dev.write(':Pulse{}:CMODe DCYCLe'.format(ch))
        self.dev.write(':PULSE{}:PCOunter {}'.format(ch,on))
        self.dev.write(':PULSE{}:OCOunter {}'.format(ch,off))
        
    @sleep_method
    @robust
    def lev(self,*p):    # set the output amplitude of a channel
        if bool(p) == True:  # adjustbale
            __ch = p[0]
            if len(p) > 1:
                __volt = p[1]
                __volt = self.rd(__volt)
                self.dev.write(':PULSE{}:OUTPut:MODe ADJustable'.format(__ch))
                self.dev.write(':PULSE{}:OUTPut:AMPLitude {}'.format(__ch,__volt))
            else:
                self.dev.write(':PULSE{}:OUTPut:MODe TTL'.format(__ch))
        else:   # TTL
            for __ch in range(1,9): 
                self.lev(__ch)
                
    @sleep_method
    @robust
    def mux(self,*p):  ## multiplexer
        if bool(p) == True:
            __ch = p[0]
            __timer = p[1::]
            __binary = 0
            for t in __timer:
                __binary = __binary + 2**(t-1)
            self.dev.write(':PULSE{}:MUX {}'.format(__ch,__binary))
        else:
            self.mux_reset()
            
    @sleep_method
    @robust
    def mux_reset(self):   # reset multiplexer
        for n in range(1,9): 
            self.mux(n,n)
            
    @sleep_method
    @robust
    def on(self,*ch):
        if bool(ch) == True:
            for channel in ch:
                self.dev.write(':PULSE{}:STAT ON'.format(channel))
        else:
            self.dev.write(':PULSE0:STAT ON')
            
    @sleep_method
    @robust
    def off(self,*ch):
        if bool(ch) == True:
            for channel in ch:
                self.dev.write(':PULSE{}:STAT OFF'.format(channel))
        else:
            self.dev.write(':PULSE0:STAT OFF')

    @robust
    def trigOn(self):  # system mode: triggered
        self.off()
        self.dev.write(':PULSE0:TRIG:MOD TRIG')  # trig enabled 
        self.on()
        self.dev.write('*TRG')  # software trigger

    @sleep_method
    @robust
    def cw(self): # continuous running mode
        self.dev.write(':PULSE0:TRIG:MOD DIS')  # trig disabled 
        
    @sleep_method
    @robust
    def trigOff(self):
        self.off()
        self.dev.write(':PULSE0:TRIG:MOD TRIG')  # trig enabled 
    
    @sleep_method
    @robust
    def high(self,*ch):  ## keep output constantly at +5V
        for c in ch:
            self.dev.write(':PULSE{}:POL INV'.format(c))
            self.off(c)
            
    @sleep_method
    @robust
    def low(self,*ch):  ## keep output constantly at 0V
        for c in ch:
            self.dev.write(':PULSE{}:POL NORM'.format(c))
            self.off(c)
            
    @sleep_method
    @robust
    def __exp(self,T0,ch,pol,tExp,tPls,tDly,nPls,nDly):  # experiment mode
        T0 = self.rd(T0)
        tExp = self.rd(tExp)
        tPls = self.rd(tPls)
        tDly = self.rd(tDly)
        __dcycl_on = nPls
        __dcycl_tot = round(tExp/T0)
        __dcycl_off = __dcycl_tot - __dcycl_on
        self.wid(ch,tPls)   ## pulse width
        self.dly(ch,tDly)
        self.pol(ch,pol)
        self.dcycl(ch,__dcycl_on,__dcycl_off)
        self.wcount(ch,nDly)  
        self.on(ch)
        
    @sleep_method
    @robust
    def config(self,cfg):   # preset configuration
        if re.search('(cal)', cfg, re.IGNORECASE)!= None:   # cavity calibration
            self.mux_reset()
            self.high(1,3) # A: cooling & B field (Off); C: Probe (On)
            self.low(2,4,8)  # B: Repum (Off); D: Control (Off); H: TTL (Off)
        elif re.search('(mot)', cfg, re.IGNORECASE)!= None:  # MOT
            self.mux_reset()
            self.low(1,3,4) 
            self.high(2)
        elif re.search('(off)', cfg, re.IGNORECASE)!= None:  # all off
            self.mux_reset()
            self.off(0,1,2,3,4,5,6,7,8)
            
    @sleep_method
    @robust
    def burst(self, ch, n_pulses):
        self.dev.write(':PULSe{}:CMOD BURS'.format(ch))
        self.dev.write(':PULSe{}:BCOunter {}'.format(ch,n_pulses))

    @sleep_method
    @robust
    def DC(self, ch, offset):
        self.high(ch)
        self.lev(ch, offset)


#================================================================
# Agilent ESG Signal Generator Family
#================================================================
    
class Agilent_ESG_SG:
    def __init__(self,address): #'GPIB::19' 
        self.address = address            
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.dev.write_termination='\n'
        self.dev.read_termination='\n'
    
    @robust
    def query(self):
        print(self.dev.query('OUTput?'))
        return
    
    @robust
    def off(self):
        self.dev.write(':OUTput OFF')
    
    @robust        
    def on(self):
        self.dev.write(':OUTput ON')  

    @robust        
    def freq(self,f): #define frequency and unit (Hz,kHz,MHz,GHz...)
        #self.dev.write(':FREQuency {} {}'.format(f,unit)) #for general units
        self.dev.write(':FREQuency {} MHz'.format(f))

    @robust
    def lev(self,amplitude):  #define ampitude and unit {dBm,dBUV,V,VEMF}
        #self.dev.write('POWer:AMPLitude {} {}'.format(amp,unit))
        self.dev.write('POWer:AMPLitude {} dBm'.format(amplitude))
        
    @robust
    def offset(self,amplitude,unit):  #define offset and unit {dBm,dBUV,V,VEMF}
        self.dev.write(':POWer:OFFSet {} {}'.format(amplitude,unit))

    @robust
    def phase(self,phase,unit): #define phase and unit {radian,degrees}
        self.dev.write(':PHASe {} {}'.format(phase,unit))

#================================================================
# Tektronix AFG3000 Series Arbitrary Function Generator
#================================================================

class tektronix_AFG3000:
    def __init__(self,address):
        self.address = address 
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        
    @robust
    def reset(self): #Reset
        self.dev.write("*RST")
        
    @robust
    def off(self):  # default: both
        self.dev.write(':OUTPut OFF')
            
    @robust
    def on(self): # default: both 
        self.dev.write(':OUTPut ON')

    @robust
    def freq(self,ch,f):
        self.dev.write('SOURce{}:FREQuency:FIXed {}'.format(ch,self.__Hz(f)));

    @staticmethod
    def __Hz(f): 
        """
        Description: Sets all frequencies to MHz Unit

        Input: f : frequency : float

        Output: f * 1e6 : frequency in MHz : float
        """
        if type(f) == str:
            if re.search('[mM]',f) != None:
                return 1e6*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[kK]',f) != None:
                return 1e3*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[hH]',f) != None:
                return 1*float(re.sub('[a-zA-Z]','',f))
        else: # float, or str that contains only numbers
            return float(f)*1e6

    @robust
    def lev(self,ch,v):
        if type(v) == str:
            __V = float(re.sub('[a-zA-Z]','',v))  # unitless value
            if re.search('[rR]',v) != None:  #  VRMS
                self.dev.write('SOURCe:VOLTage:UNIT VRMS'.format(ch))
            elif re.search('[dD]',v) != None:  # dBm
                self.dev.write('SOURCe:VOLTage:UNIT DBM'.format(ch))
        else:  # default: [Vpp] 
            __V = v
            self.dev.write(':SOURCe{}:VOLTage:UNIT VPP'.format(ch))

        self.dev.write('SOURCe{}:VOLTage {}'.format(ch,__V))
        self.dev.write('SOURCe{}:VOLTage:UNIT VPP')

    @robust
    def offset(self,ch,offset):  # V_DC
        self.dev.write('SOURce{}:VOLTage:LEVel:IMMediate:OFFSet {}'.format(ch,offset));

    @robust
    def phase(self,ch,phase):
        self.dev.write('SOURce{}:PHASe {}'.format(ch,phase));
        
    @robust
    def burst_delay(self,ch,tdelay):
        self.dev.write('SOURce{}:BURS:TDEL {}ns'.format(ch,tdelay))
    
    @staticmethod
    def gaussian(t,mu,FWHM,a): #Gaussian Function. Inputs: (Center, FWHM, Amplitude)
        sigma = (FWHM)/(2*np.sqrt(2*np.log(2)))
        amplitude = np.sqrt(2*np.pi*sigma**2)*a
        return amplitude*( 1/(sigma * np.sqrt(2*np.pi) ) )*np.exp( -((t-mu)**2 / (2*sigma**2)) )
    
    @staticmethod
    def square(t,leadingedge,width,amp): #square pulse with duty cycle
        return np.piecewise(t,[(t<=leadingedge),((t>leadingedge) & (t<leadingedge+width)),(t>=leadingedge+width)],[0,amp,0])
    
    @staticmethod
    def normalize(waveform):
        """
        Description: Normalizes data for arbitrary waveform design, points are limited
        to -1 to 1 Volt

        Input: data : waveform : np.array or list

        Output: np.array(waveform)/np.absolute(max(waveform)) : normalized data :  np.array or list
        """
        factor = max([np.abs(max(waveform)),np.abs(min(waveform))])
        return np.array(waveform)/np.absolute(factor)
    
    @robust
    def arbitrary(self, ch, signal_width, waveform, *arg):
        """
        Description: Allows one to set and create arbitrary waveform output 

        Input: signal_width : width (time) of the argument in seconds : float
               waveform : method or list of values : method or array
               arg* : arguments of the waveform function : misc

        Output: None : class method
        """
        buffer_size = 2**14-2
        
        
        if inspect.ismethod(waveform) == True or inspect.isfunction(waveform) == True:

            t = np.linspace(0,signal_width,buffer_size)
            data = waveform(t,*arg)
            datastring = self.normalize(data)
            
        else:
            
            data = waveform
            datastring = self.normalize(data)
            
        if (datastring.max() - datastring.min()) == 0:
            raise ValueError('Use the DC-function')
        m = buffer_size / (datastring.max() - datastring.min())
        b = -m * datastring.min()
        dac_values = (m * datastring + b)
        np.around(dac_values, out=dac_values)
        dac_values = dac_values.astype(np.uint16)
            
        self.dev.write('DATA:DEFine EMEMory,{}'.format(len(data)))
        self.dev.write_binary_values("DATA:DATA EMEM1,", dac_values, datatype="H", is_big_endian=True)
        self.dev.write("SOURce{}:FUNC:SHAPE EMEM1".format(ch))
        self.dev.write("SOURce{}:VOLTage:LEVel:IMMediate:LOW {}".format(ch,min(data)))
        self.dev.write("SOURce{}:VOLTage:LEVel:IMMediate:HIGH {}".format(ch,max(data)))
        self.dev.write("SOURce{}:FREQuency:FIXED {}".format(ch, 1/signal_width))

    @robust
    def ext_trig(self):
        self.dev.write("TRIGger:SEQuence:SOURce EXTernal")
    
    @robust
    def burst_mode(self, ch, mode, cycles):
        """
        Description: Allows on to use burst functionallity

        Input: mode : burst mode : int = {'TRIG' = 0, 'GAT' = 1}
               cycles : cycles : int

        Output: 
        """
        modes = ['TRIG','GAT']
        self.dev.write("SOURCE{}:BURSt:MODE {}".format(ch,modes[mode]))
        self.dev.write("SOURCE{}:BURSt:NCYC {}".format(ch,cycles))
        self.dev.write("SOURCE{}:BURSt:STAT ON".format(ch))
    
    @robust
    def burst_state(self, ch, state):
        states = ['OFF','ON']
        self.dev.write('SOURce{}:BURSt:STAT {}'.format(ch, states[state]))
    
    @robust
    def DC(self, ch, offset):
        """
        Description: Enables DC Mode (Tested 04/03/2022)

        Input: ch : channel : int
               offset : DC Offset : float

        Output: None : class method
        """
        self.dev.write("SOUR{}:FUNC:SHAP DC".format(ch))
        self.dev.write("SOUR{}:VOLT:LEV:IMM:OFFS {}".format(ch,offset))

#================================================================
# Rigol DP8000 Series DC Power Supply
#================================================================        

class Rigol_DP800:

    def __init__(self,address): 
        """ 
        Description: This function will initialize the power supply class object 
    
        Input: IP address : string
        Output: power supply object with built in funtions : class object

        Example: 
        >>gen1 = gen.Rigol_DP800('TCPIP::10.2.1.158')

        """
        self.address = address            
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
    
    @robust
    def inspect(self): 
        """ 
        Description: This function will return to you the current settings of the 
        power supply. This includes overvoltage/overcurrent protection, overvoltage/current protection value 
        state on each channel, range of each channel
    
    
        Input: None : None
        Output: Characterization List : String

        Example: 
        >>gen1.inspect()
        
        (CH1 :: Mode : CV :: OCP state : ON :: OCP value : 0.5mA ::  OVP state : OFF :: OVP value : 0.5V :: Range : 10A/5V :: State : ON)
        etc...

        """
        try: 
            for i in range(1,4):
                ch = 'CH{} :: '.format(i)
                ch = ch,'Mode : ',str(self.dev.query(':OUTP:MODE? CH{};*OPC?'.format(i))),' :: '
                ch = ch,'OCP state : ',str(self.dev.query(':OUTP:OCP? CH{};*OPC?'.format(i))),' :: '
                ch = ch,'OCP value : ',str(self.dev.query(':OUTP:OCP:VAL? CH{};*OPC?'.format(i))),' :: '
                ch = ch,'OVP state : ',str(self.dev.query(':OUTP:OVP? CH{};*OPC?'.format(i))),' :: '
                ch = ch,'OVP value : ',str(self.dev.query(':OUTP:OVP:VAL? CH{};*OPC?'.format(i))),' :: '
                print(ch)
        except:
            print('unable to obtain parameters')
        return
    
    @robust
    def set_safety(self): 
        """ 
        Description: This function will return to you the current settings of the 
        power supply. This includes overvoltage/overcurrent protection, overvoltage/current protection value 
        state on each channel, range of each channel
    
    
        Input: None : None
        Output: Characterization List : String

        Example: 
        >>gen1.set_safety()

        """
        try: 
            for i in range(1,4):
                self.dev.write(':OUTP:OCP CH{},ON'.format(i))
                self.dev.write(':OUTP:OVP CH{},ON'.format(i))
            print('SAFETY SET')
        except:
            print('unable to excecute')
        return
    
    @robust
    def on(self, *ch): 
        """ 
        Description: Will turn on specified channel
    
    
        Input: ch : [1,2,3]
        Output: Method which turns the output state to on

        Example: 
        >>gen1.on(1)

        """
        if bool(ch) == True:
            self.dev.write(':OUTP CH{}, ON'.format(ch[0]))
        else:
            for i in range(1,4):
                self.dev.write(':OUTP CH{}, ON'.format(i))
        return
    
    @robust
    def off(self, *ch): 
        """ 
        Description: Will turn off specified channel
    
    
        Input: ch : [1,2,3]
        Output: Method which turns the output state to off

        Example: 
        >>gen1.off(2)

        """
        if bool(ch) == True:
            self.dev.write(':OUTP CH{}, OFF'.format(ch[0]))
        else:
            for i in range(1,4):
                self.dev.write(':OUTP CH{}, OFF'.format(i))
        return
    
    @robust
    def current(self, ch, *current): 
        """ 
        Description: This function will set or get the current of the specified channel
    
    
        Input: channel : [1,2,3], current : float
        Output: Method which changes the current

        Example: 
        >>gen1.current(1,2)

        """
        if current:
            self.dev.write('SOUR{}:CURR {}'.format(ch,current[0])) #Units Amps
        else:
            self.dev.write('SOUR{}:CURR?'.format(ch))
        return

    @robust
    def voltage(self, ch, *voltage): 
        """ 
        Description: This function will set or get the voltage of the specified channel
    
    
        Input: channel : [1,2,3], voltage : float
        Output: Method which changes the voltage

        Example: 
        >>gen1.voltage(1,2)

        """
        if bool(voltage) == True:
            self.dev.write('SOUR{}:VOLT {}'.format(ch,voltage[0])) #Units Volts
        else:
            print(self.dev.query('SOUR{}:VOLT?'.format(ch)))
        return