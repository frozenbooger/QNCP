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
   
    def off(self): 
        self.write('OEN OFF;')  # turn off RF output
        self.write('PDN OFF;')  # turn off synth output
        
    def on(self):
        self.write('OEN ON;')  # turn on RF output
        self.write('PDN ON;');  # turn on synth output

    def freq(self,*f):  # MHz, enter or read frequency
        if bool(f) == True:   # assign value (only the first in the tuple)
            if type(f[0]) == str:
                self.write('Freq {};'.format(f[0]));
            else:
                self.write('Freq {} MHz;'.format(f[0]));
        else:   # when no input is entered, read actual frequency 
            self.write('Freq?;')
            self.__readOut = self.read()
            self.__freqAct = float(re.search('(?<=Act )\S+',self.__readOut).group())
            return self.__freqAct

    def lev(self,*l):  # MHz
        if bool(l) == True:  # assign level
            if type(l[0]) == str:
                self.__pwr = float(re.search('[0-9]*(\.)*[0-9]*',l[0]).group())
                self.write('PWR {};'.format(self.__pwr));
            else:
                self.write('PWR {};'.format(l[0]));
        else:  # when empty input, read actual level
            self.write('PWR?;')
            self.__readOut = self.read()
            self.__levAct = float(re.search('(?<=PWR ).*(?=\;)',self.__readOut).group())
            return self.__levAct
        
    def write(self,arg):
        self.clear()    
        self.dev.write(arg)
        
    def read(self):
        return self.dev.read()

    def clear(self):  # clear the device command. Very important for Valon!
        self.dev.clear()
#     def close(self):
#         self.dev.close()

#================================================================
# Function Generator - Rigol DSG800 series
#================================================================

class Rigol_DSG800:
    
    def __init__(self,address,*arg):
        if arg:
            self.address = address #'TCPIP::<IP ADDRESS>::INSTR'
            self.dev = vxi11.Instrument(self.address)
        else:
            self.address = address 
            self.rm = pyvisa.ResourceManager()
            self.dev = self.rm.open_resource(self.address)
     
    def off(self):
        self.dev.write(':OUTput OFF;')  # turn on RF output
        
    def on(self):
        self.dev.write(':OUTput On;')  # turn on RF output
    
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
        
    def reset(self): #Reset
        self.dev.write("*RST")
        
    def off(self,*ch):  # default: both
        if bool(ch) == True: # turn single output off
            for channel in ch:
                self.dev.write(':OUTput{} OFF'.format(channel))  
        else:  # turn both off
            self.dev.write(':OUTput1 OFF') 
            self.dev.write(':OUTput2 OFF')
    def on(self,*ch): # default: both
        if bool(ch) == True: # turn single output on
            for channel in ch:
                self.dev.write(':OUTput{} ON'.format(channel))  
        else:  # turn both no
            self.dev.write(':OUTput1 ON') 
            self.dev.write(':OUTput2 ON')

    def __Hz(f):  # in Hz, support unit. Default: MHz
        """
        Description: Sets all frequencies to MHz Unit (Tested 04/03/2022)

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

    def freq(self,ch,*f):
        if bool(f) == True:   # assign value (only the first in the tuple)
            __freq = f[0]
            self.dev.write(':SOURCe{}:Freq {}'.format(ch,self.__Hz(__freq)));
        else:   # when no input is entered, read actual frequency 
            __readOut = self.dev.query(':SOURCe{}:Freq?'.format(ch))
            __freq = float(re.search('.*(?=\n)',__readOut).group())
            return float(__freq)*1e-6  # MHz
        
    def lev(self,ch,*v):
        if bool(v) == True:
            __v = v[0]
            if type(__v) == str:
                __lev = float(re.sub('[a-zA-Z]','',__v))  # unitless value
                if re.search('(vrms)',__v,re.IGNORECASE) != None:  #  VRMS
                    self.dev.write(':SOURCe{}:VOLTage:UNIT VRMS'.format(ch))
                elif re.search('d',__v,re.IGNORECASE) != None:  # dBm
                    self.dev.write(':SOURCe{}:VOLTage:UNIT DBM'.format(ch))
                else:  # VPP
                    self.dev.write(':SOURCe{}:VOLTage:UNIT VPP'.format(ch))
                # mVPP or mVRMS
                if re.search('(mv)',__v,re.IGNORECASE) != None:
                    __lev = 1e-3 * __lev
                # value
                self.dev.write(':SOURCe{}:VOLTage {}'.format(ch,__lev))
            else:  # default: [Vpp] 
                self.dev.write(':SOURCe{}:VOLTage:UNIT VPP'.format(ch))
                self.dev.write(':SOURCe{}:VOLTage {}'.format(ch,__v))
        else:
            __readOut = self.dev.query(':SOURCe{}:VOLTage?'.format(ch))
            __lev = float(re.search('.*(?=\n)',__readOut).group())
            __readOut = self.dev.query(':SOURCe{}:VOLTage:UNIT?'.format(ch))
            __unit = re.search('.*(?=\n)',__readOut).group()
            return __lev, __unit

    def offset(self,ch,offset):  # V_DC
        self.dev.write(':SOURCe{}:VOLTage:OFFSet {}'.format(ch,offset));

    def phase(self,ch,p):
        self.dev.write(':SOURCe{}:PHASe {}'.format(ch,p));
        
    def gaussian(self,t,mu,FWHM,a): #Gaussian Function. Inputs: (FWHM, Amplitude, Center)
        sigma = (FWHM)/(2*np.sqrt(2*np.log(2)))
        amplitude = np.sqrt(2*np.pi*sigma**2)*a
        return amplitude*( 1/(sigma * np.sqrt(2*np.pi) ) )*np.exp( -((t-mu)**2 / (2*sigma**2)) )
    
    def square(self,t,leadingedge,width,amp): #square pulse with duty cycle
        return np.piecewise(t,[(t<=leadingedge),((t>leadingedge) & (t<leadingedge+width)),(t>=leadingedge+width)],[0,amp,0])
    
    def normalize(self,waveform):
        """
        Description: Normalizes data for arbitrary waveform design, points are limited (Tested 04/03/2022)
        to -1 to 1 Volt

        Input: data : waveform : np.array or list

        Output: np.array(waveform)/np.absolute(max(waveform)) : normalized data :  np.array or list
        """
        factor = max([np.abs(max(waveform)),np.abs(min(waveform))])
        return np.array(waveform)/np.absolute(factor)
        
    def arbitrary(self, ch, signal_width, waveform, *arg):
        """
        Description: Allows one to set and create arbitrary waveform output (Tested 04/03/2022) 

        Input: ch : channel : int
               signal_width : width (time) of the argument in seconds : float
               waveform : method or list of values : method or array
               arg* : arguments of the waveform function : misc

        Output: None : class method
        """
        buffer_size = 2**14
        if inspect.ismethod(waveform) == True:
            t = np.linspace(0,signal_width,buffer_size)
            data = np.around(waveform(t,*arg),4)
            datastring = ",".join(map(str,self.normalize(data)))
        else:
            data = np.around(waveform,4)
            datastring = ",".join(map(str,self.normalize(data)))
        
        factor = max([np.abs(max(data)),np.abs(min(data))])
        self.dev.write('SOURCE{}:Freq {}'.format(ch, 1/signal_width))
        self.dev.write("SOURCE{}:TRACE:DATA VOLATILE,".format(ch) + datastring)
        self.dev.write("SOURCE{}:FUNC VOLATILE,".format(ch))
        self.dev.write("SOURCE{}:VOLTAGE:LOW {}".format(ch,-factor))
        self.dev.write("SOURCE{}:VOLTAGE:HIGH {}".format(ch,factor))
        self.dev.write("SOURCE{}:PHASE:SYNC".format(ch))
    
    def burst(self, ch, mode, cycles):
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
        
    def DC(self, ch, offset):
        """
        Description: Enables DC Mode (Tested 04/03/2022)

        Input: ch : channel : int
               offset : DC Offset : float

        Output: None : class method
        """
        self.dev.write("SOURCE{}:FUNC DC".format(ch))
        self.offset(ch,offset)
    
    def ext_trig(self,ch):
        self.dev.write("SOURCE{}:BURST:TRIG:SOUR EXT".format(ch))
        
    def arbitrary_burst(self,ch,signal_width,cycles,func,*arg):
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
        if inspect.ismethod(func) == True:
            t = np.linspace(0,signal_width,1000)
            data = func(t,*arg)
            datastring = ",".join(map(str, self.normalize(data)))
        else:
            data = func
            datastring = ",".join(map(str, self.normalize(data)))

        self.dev.write('SOURCE{}:Freq {}'.format(ch, 1/signal_width))
        self.dev.write("OUTPUT{} ON".format(ch))
        self.dev.write("SOURCE{}:TRACE:DATA VOLATILE,".format(ch)+ datastring)
        self.dev.write("SOURCE{}:VOLTAGE:UNIT VPP".format(ch))
        self.dev.write("SOURCE{}:VOLTAGE:AMPL {}".format(ch,2*max(data)))
        # self.dev.write("SOURCE{}:VOLTAGE:LOW {}".format(ch,min(data)))
        # self.dev.write("SOURCE{}:VOLTAGE:HIGH {}".format(ch,max(data)))
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
class MOGDevice:
    def __init__(self,addr,port=None,timeout=1,check=True):
        # is it a COM port?
        if addr.startswith('/dev/tty.') or addr == 'USB':
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
    
    def reconnect(self,timeout=1,check=True):
        "Reestablish connection with unit"
        if hasattr(self,'dev'): self.dev.close()
        if self.is_usb:
            import serial
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
                self.info = self.ask(b'info')
            except Exception as E:
                logger.error(str(E))
                raise RuntimeError('Device did not respond to query')
    
    def versions(self):
        verstr = self.ask(b'version')
        if verstr == b'Command not defined':
            raise RuntimeError('Incompatible firmware')
        # does the version string define components?
        vers = {}
        if b':' in verstr:
            # old versions are LF-separated, new are comma-separated
            tk = b',' if b',' in verstr else '\n'
            for l in verstr.split(tk):
                if l.startswith(b'OK'): continue
                n,v = l.split(b':',2)
                v = v.strip()
                if b' ' in v: v = v.rsplit(' ',2)[1].strip()
                vers[n.strip()] = v
        else:
            # just the micro
            vers[b'UC'] = verstr.strip()
        return vers

    def cmd(self,cmd):
        "Send the specified command, and check the response is OK"
        resp = self.ask(cmd)
        if resp.startswith(b'OK'):
            return resp
        else:
            raise RuntimeError(resp)
        
    def ask(self,cmd):
        "Send followed by receive"
        # check if there's any response waiting on the line
        self.flush()
        self.send(cmd)
        resp = self.recv().strip()
        if resp.startswith(b'ERR:'):
            raise RuntimeError(resp[4:].strip())
        return resp
        
    def ask_dict(self,cmd):
        "Send a request which returns a dictionary response"
        resp = self.ask(cmd)
        # might start with "OK"
        if resp.startswith(b'OK'): resp = resp[3:].strip()
        # expect a colon in there
        if not b':' in resp: raise RuntimeError('Response to '+repr(cmd)+' not a dictionary')
        # response could be comma-delimited (new) or newline-delimited (old)
        vals = OrderedDict()
        for entry in resp.split(b',' if b',' in resp else b'\n'):
            name, val = entry.split(b':')
            vals[name.strip()] = val.strip()
        return vals
        
    def ask_bin(self,cmd):
        "Send a request which returns a binary response"
        self.send(cmd)
        head = self.recv_raw(4)
        # is it an error message?
        if head == b'ERR:': raise RuntimeError(self.recv().strip())
        datalen = unpack('<L',head)[0]
        data = self.recv_raw(datalen)
        if len(data) != datalen: raise RuntimeError('Binary response block has incorrect length')
        return data
    
    def send(self,cmd):
        "Send command, appending newline if not present"
        if hasattr(cmd,'encode'):  cmd = cmd.encode()
        if not cmd.endswith(CRLF): cmd += CRLF
        self.send_raw(cmd)
    
    def has_data(self,timeout=0):
        if self.is_usb:
            import serial
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
        dat = b''
        while self.has_data(timeout):
            dat += self.recv(buffer)
        if len(dat): logger.debug('Flushed'+repr(dat))
        return dat
    
    def recv(self,buffer=256):
        "A somewhat robust multi-packet receive call"
        if self.is_usb:
            data = self.dev.readline(buffer)
            if len(data):
                t0 = self.dev.timeout
                self.dev.timeout = 0 if data.endswith(CRLF) else 0.1
                while True:
                    segment = self.dev.readline(buffer)
                    if len(segment) == 0: break
                    data += segment
                self.dev.timeout = t0
            if len(data) == 0: raise RuntimeError('Timed out')
        else:
            data = self.dev.recv(buffer)
            timeout = 0 if data.endswith(CRLF) else 0.1
            while self.has_data(timeout):
                try:
                    segment = self.dev.recv(buffer)
                except IOError:
                    if len(data): break
                    raise
                data += segment
        logger.debug('<< %d = %s'%(len(data),repr(data)))
        return data
    
    def send_raw(self,cmd):
        "Send, without appending newline"
        if len(cmd) < 256:
            logger.debug('>>'+repr(cmd))
        if self.is_usb:
            return self.dev.write(cmd)
        else:
            return self.dev.send(cmd)
    
    def recv_raw(self,size):
        "Receive exactly 'size' bytes"
        # be pythonic: better to join a list of strings than append each iteration
        parts = []
        while size > 0:
            if self.is_usb:
                chunk = self.dev.read(size)
            else:
                chunk = self.dev.recv(size)
            if len(chunk) == 0:
                break
            parts.append(chunk)
            size -= len(chunk)
        buf = b''.join(parts)
        logger.debug('<< RECV_RAW got %d'%len(buf))
        logger.debug(repr(buf))
        return buf
        
    def set_timeout(self,val = None):
        if self.is_usb:
            old = self.dev.timeout
            if val is not None: self.dev.timeout = val
            return old
        else:
            old = self.dev.gettimeout()
            if val is not None: self.dev.settimeout(val)
            return old

# Driver
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

    def freq(self,ch,frequency):
        if type(f) == str:
            self.dev.cmd('FREQ,{},{}'.format(ch,frequency))
        else:
            self.dev.cmd('FREQ,{},{} MHz'.format(ch,frequency))

    def lev(self,ch,amplitude):
        if type(amplitude) == str:
            self.dev.cmd('POW,{},{}'.format(ch,amplitude))
        else:
            self.dev.cmd('POW,{},{} dBm'.format(ch,amplitude))

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
            self.address = address
            self.baud_rate = arg[0]
            self.rm = pyvisa.ResourceManager()
            self.dev = self.rm.open_resource(address, # same as 'ASRL5::INSTR'
                                  baud_rate = self.baud_rate, # must identify
                                  data_bits = 8,
                                  parity = Parity.none,
                                  stop_bits = StopBits.one)
            self.mux_reset()   # clear all multiplexer
            self.lev()    # set all outputs to TTL
            self.t_sleep = 50e-3
            self.digit   = 11   # important! round evertying to 11 digits
        else:
            self.address = address
            self.rm = pyvisa.ResourceManager()
            self.dev = self.rm.open_resource(address)
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
    def t0(self,t):  # clock T0
        t = self.rd(t)
        self.dev.write(':PULSE0:PER {}'.format(t))
        
    @sleep_method
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
    def wid(self,ch,w):
        w = self.rd(w)
        self.dev.write(':PULSE{}:WIDth {}'.format(ch,w))
        
    @sleep_method
    def dly(self,ch,d):
        d = self.rd(d)
        self.dev.write(':PULSE{}:DELay {}'.format(ch,d))
        
    @sleep_method
    def pol(self,ch,p):
        self.dev.write(':PULSE{}:POL {}'.format(ch,p))
        
    @sleep_method
    def wcount(self,ch,w):  # wait number of T0 before enable output       
        self.dev.write(':PULSE{}:WCOunter {}'.format(ch,w))
        
    @sleep_method
    def dcycl(self,ch,on,off):   # channel duty cycle
        self.dev.write(':Pulse{}:CMODe DCYCLe'.format(ch))
        self.dev.write(':PULSE{}:PCOunter {}'.format(ch,on))
        self.dev.write(':PULSE{}:OCOunter {}'.format(ch,off))
        
    @sleep_method
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
    def mux_reset(self):   # reset multiplexer
        for n in range(1,9): 
            self.mux(n,n)
            
    @sleep_method
    def on(self,*ch):
        if bool(ch) == True:
            for channel in ch:
                self.dev.write(':PULSE{}:STAT ON'.format(channel))
        else:
            self.dev.write(':PULSE0:STAT ON')
            
    @sleep_method
    def off(self,*ch):
        if bool(ch) == True:
            for channel in ch:
                self.dev.write(':PULSE{}:STAT OFF'.format(channel))
        else:
            self.dev.write(':PULSE0:STAT OFF')

    def trigOn(self):  # system mode: triggered
        self.off()
        self.dev.write(':PULSE0:TRIG:MOD TRIG')  # trig enabled 
        self.on()
        self.dev.write('*TRG')  # software trigger

    @sleep_method        
    def cw(self): # continuous running mode
        self.dev.write(':PULSE0:TRIG:MOD DIS')  # trig disabled 
        
    @sleep_method
    def trigOff(self):
        self.off()
        self.dev.write(':PULSE0:TRIG:MOD TRIG')  # trig enabled 
    @sleep_method
    def high(self,*ch):  ## keep output constantly at +5V
        for c in ch:
            self.dev.write(':PULSE{}:POL INV'.format(c))
            self.off(c)
            
    @sleep_method
    def low(self,*ch):  ## keep output constantly at 0V
        for c in ch:
            self.dev.write(':PULSE{}:POL NORM'.format(c))
            self.off(c)
            
    @sleep_method   
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
    def burst(self, ch, n_pulses):
        self.dev.write(':PULSe{}:CMOD BURS'.format(ch))
        self.dev.write(':PULSe{}:BCOunter {}'.format(ch,n_pulses))

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
        
    def query(self):
        print(self.dev.query('OUTput?'))
        return
     
    def off(self):
        if int(self.dev.query('OUTput?')) == 1: # turn single output off
            self.dev.write(':OUTput OFF')  
        else:  # it is already off
            print('I am already off')
            
    def on(self):
        if int(self.dev.query('OUTput?')) == 0: # turn single output off
            self.dev.write(':OUTput ON')  
        else:  # it is already off
            print('I am already on')

    def freq(self,f): #define frequency and unit (Hz,kHz,MHz,GHz...)
        #self.dev.write(':FREQuency {} {}'.format(f,unit)) #for general units
        self.dev.write(':FREQuency {} MHz'.format(f))

    def lev(self,amplitude):  #define ampitude and unit {dBm,dBUV,V,VEMF}
        #self.dev.write('POWer:AMPLitude {} {}'.format(amp,unit))
        self.dev.write('POWer:AMPLitude {} dBm'.format(amplitude))
        
    def offset(self,amplitude,unit):  #define offset and unit {dBm,dBUV,V,VEMF}
        self.dev.write(':POWer:OFFSet {} {}'.format(amplitude,unit))

    def phase(self,phase,unit): #define phase and unit {radian,degrees}
        self.dev.write(':PHASe {} {}'.format(phase,unit))

#================================================================
# Tektronix AFG3000 Series Arbitrary Function Generator
#================================================================

class tektronix_AFG3000:
    def __init__(self,address,*arg):
        self.address = address 
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        
    def reset(self): #Reset
        self.dev.write("*RST")
        
    def off(self):  # default: both
        self.dev.write(':OUTPut OFF')
            
    def on(self): # default: both 
        self.dev.write(':OUTPut ON')

    def freq(self,f):
        self.dev.write('SOURce:FREQuency:FIXed {}'.format(self.__Hz(f)));

    def __Hz(self, f): 
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

    def lev(self,v):
        if type(v) == str:
            __V = float(re.sub('[a-zA-Z]','',v))  # unitless value
            if re.search('[rR]',v) != None:  #  VRMS
                self.dev.write('SOURCe:VOLTage:UNIT VRMS')
            elif re.search('[dD]',v) != None:  # dBm
                self.dev.write('SOURCe:VOLTage:UNIT DBM')
        else:  # default: [Vpp] 
            __V = v
            self.dev.write(':SOURCe:VOLTage:UNIT VPP')

        self.dev.write('SOURCe:VOLTage {}'.format(__V))
        self.dev.write('SOURCe:VOLTage:UNIT VPP')

    def offset(self,offset):  # V_DC
        self.dev.write('SOURC:VOLTage:OFFSet {}'.format(offset));

    def phase(self,phase):
        self.dev.write('SOURC:PHASe {}'.format(phase));
        
    def burst_delay(self,tdelay):
        self.dev.write('SOUR:BURS:TDEL {}ns'.format(tdelay))
        
    def gaussian(self,t,mu,FWHM,a): #Gaussian Function. Inputs: (FWHM, Amplitude, Center)
        sigma = (FWHM)/(2*np.sqrt(2*np.log(2)))
        amplitude = np.sqrt(2*np.pi*sigma**2)*a
        return amplitude*( 1/(sigma * np.sqrt(2*np.pi) ) )*np.exp( -((t-mu)**2 / (2*sigma**2)) )
    
    def square(self,t,leadingedge,width,amp): #square pulse with duty cycle
        return np.piecewise(t,[(t<=leadingedge),((t>leadingedge) & (t<leadingedge+width)),(t>=leadingedge+width)],[0,amp,0])
    
    def normalize(self, waveform):
        """
        Description: Normalizes data for arbitrary waveform design, points are limited
        to -1 to 1 Volt

        Input: data : waveform : np.array or list

        Output: np.array(waveform)/np.absolute(max(waveform)) : normalized data :  np.array or list
        """
        factor = max([np.abs(max(waveform)),np.abs(min(waveform))])
        return np.array(waveform)/np.absolute(factor)
    
    def arbitrary(self, signal_width, waveform, *arg):
        """
        Description: Allows one to set and create arbitrary waveform output 

        Input: signal_width : width (time) of the argument in seconds : float
               waveform : method or list of values : method or array
               arg* : arguments of the waveform function : misc

        Output: None : class method
        """
        buffer_size = 2**14-2
        
        if waveform[0] != waveform[-1]:
            waveform[0] = 0
            waveform[-1] = waveform[0]
        else:
            pass
        
        if inspect.ismethod(waveform) == True:

            t = np.linspace(0,total_time,buffer_size)
            data = waveform(t,*arg)

            datastring = self.normalize(data)
            m = buffer_size / (datastring.max() - datastring.min())
            b = -m * datastring.min()
            dac_values = (m * datastring + b)
            np.around(dac_values, out=dac_values)
            dac_values = dac_values.astype(np.uint16)

        else:
            data = waveform
            datastring = self.normalize(data)
            m = buffer_size / (datastring.max() - datastring.min())
            b = -m * datastring.min()
            dac_values = (m * datastring + b)
            np.around(dac_values, out=dac_values)
            dac_values = dac_values.astype(np.uint16)    
            
        factor = max([np.abs(max(data)),np.abs(min(data))])
        self.dev.write('DATA:DEFine EMEMory,{}'.format(len(data)))
        self.dev.write_binary_values("DATA:DATA EMEM1,", dac_values, datatype="H", is_big_endian=True)
        self.dev.write("SOURce:FUNC:SHAPE EMEM1")
        self.dev.write("SOURce1:VOLTage:LEVel:IMMediate:LOW {}".format(-factor))
        self.dev.write("SOURce1:VOLTage:LEVel:IMMediate:HIGH {}".format(factor))
        self.dev.write("SOURCE:FREQ {}".format(self.__Hz(1/signal_width)))

    def ext_trig(self):
        self.dev.write("TRIGger:SEQuence:SOURce EXTernal")
    
    def burst(self, ch, mode, cycles):
        """
        Description: Allows on to use burst functionallity

        Input: ch : channel : int
               mode : burst mode : int = {'TRIG' = 0, 'GAT' = 1,'INF' = 2}
               cycles : cycles : int

        Output: 
        """
        modes = ['TRIGgered','GATed','INFinity']
        self.dev.write('SOURce{}:BURSt:MODE {}'.format(ch,modes[mode]))
        self.dev.write('SOURce{}:BURSt:TRIGger:SOURce EXT'.format(ch))
        self.dev.write('SOURce{}:BURSt:NCYCles {}'.format(ch,cycles))
        self.dev.write('SOURce{}:BURSt ON'.format(ch))
    
    def DC(self, offset):
        """
        Description: Enables DC Mode (Tested 04/03/2022)

        Input: ch : channel : int
               offset : DC Offset : float

        Output: None : class method
        """
        self.dev.write("SOUR:FUNC:SHAP DC")
        self.dev.write("SOUR:VOLT:LEV:IMM:OFFS {}".format(offset))
        
        

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