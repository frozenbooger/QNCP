import pyvisa
from pyvisa.constants import Parity,StopBits
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from scipy.signal import find_peaks
import inspect
import vxi11
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
# Rigol_DSG830 (tested)
#================================================================

class Rigol_DSG830:
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
# Rigol_DG4202 (1.2 tested)
#================================================================
class Rigol_DG4202:
    def __init__(self,address,*arg):
        if arg:
            self.address = address #'TCPIP::<IP ADDRESS>::INSTR'
            self.dev = vxi11.Instrument(self.address)
        else:
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

    def freq(self,ch,f):
        self.dev.write(':SOURCe{}:Freq {}'.format(ch,self.__Hz(f)));

    def __Hz(self,f):  # in Hz, support unit. Default: MHz
        if type(f) == str:
            if re.search('[mM]',f) != None:
                return 1e6*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[kK]',f) != None:
                return 1e3*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[hH]',f) != None:
                return 1*float(re.sub('[a-zA-Z]','',f))
        else: # float, or str that contains only numbers
            return float(f)*1e6

    def lev(self,ch,v):
        if type(v) == str:
            __V = float(re.sub('[a-zA-Z]','',v))  # unitless value
            if re.search('[rR]',v) != None:  #  VRMS
                self.dev.write(':SOURCe{}:VOLTage:UNIT VRMS'.format(ch))
            elif re.search('[dD]',v) != None:  # dBm
                self.dev.write(':SOURCe{}:VOLTage:UNIT DBM'.format(ch))
        else:  # default: [Vpp] 
            __V = v
            self.dev.write(':SOURCe{}:VOLTage:UNIT VPP'.format(ch))

        self.dev.write(':SOURCe{}:VOLTage {}'.format(ch,__V))
        self.dev.write(':SOURCe{}:VOLTage:UNIT VPP'.format(ch))

    def offset(self,ch,o):  # V_DC
        self.dev.write(':SOURCe{}:VOLTage:OFFSet {}'.format(ch,o));

    def phase(self,ch,p):
        self.dev.write(':SOURCe{}:PHASe {}'.format(ch,p));
        
    def gaussian(self,t,mu,FWHM,a): #Gaussian Function. Inputs: (FWHM, Amplitude, Center)
        sigma = (FWHM)/(2*np.sqrt(2*np.log(2)))
        amplitude = np.sqrt(2*np.pi*sigma**2)*a
        return amplitude*( 1/(sigma * np.sqrt(2*np.pi) ) )*np.exp( -((t-mu)**2 / (2*sigma**2)) )
    
    def square(self,t,leadingedge,width,amp): #square pulse with duty cycle
        return np.piecewise(t,[(t<=leadingedge),((t>leadingedge) & (t<leadingedge+width)),(t>=leadingedge+width)],[0,amp,0])
    
    def normalize(self,data):
        return np.array(data)/max(data)
        
    def arb(self,ch,freq,func,*arg):
        total_time = 1/(self.__Hz(freq))
        
        if inspect.ismethod(func) == True:
            t = np.linspace(0,total_time,1000)
            data = func(t,*arg)
            datastring = ",".join(map(str,self.normalize(data)))
        else:
            data = func
            datastring = ",".join(map(str,self.normalize(data)))
        
        self.dev.write("OUTPUT{} ON".format(ch))
        self.dev.write("SOURCE{}:TRACE:DATA VOLATILE,".format(ch)+ datastring)
        self.dev.write('SOURCE{}:Freq {}'.format(ch,self.__Hz(freq)))
        self.dev.write("SOURCE{}:VOLTAGE:UNIT VPP".format(ch))
        self.dev.write("SOURCE{}:VOLTAGE:AMPL {}".format(ch,max(data)*2))
#         self.dev.write("SOURCE{}:VOLTAGE:LOW {}".format(ch,min(data)))
#         self.dev.write("SOURCE{}:VOLTAGE:HIGH {}".format(ch,max(data)))
        self.dev.write("SOURCE{}:VOLTAGE:OFFSET 0".format(ch))
        self.dev.write("SOURCE{}:PHASE 0".format(ch))
        self.dev.write("SOURCE{}:PERIOD {}".format(ch,total_time))
        self.dev.write("SOURCE{}:PHASE:SYNC".format(ch))
    
    def ext_trig(self,ch):
        self.dev.write("SOURCE{}:BURST:TRIG:SOUR EXT".format(ch))
    
    def arb_burst(self,ch,freq,cycles,func,*arg):
        total_time = 1/(self.__Hz(freq))
        
        if inspect.ismethod(func) == True:
            t = np.linspace(0,total_time,1000)
            data = func(t,*arg)
            datastring = ",".join(map(str,self.normalize(data)))
        else:
            data = func
            datastring = ",".join(map(str,self.normalize(data)))
        
        self.dev.write("OUTPUT{} ON".format(ch))
        self.dev.write("SOURCE{}:TRACE:DATA VOLATILE,".format(ch)+ datastring)
        self.dev.write('SOURCE{}:Freq {}'.format(ch,self.__Hz(freq)))
        self.dev.write("SOURCE{}:VOLTAGE:UNIT VPP".format(ch))
        self.dev.write("SOURCE{}:VOLTAGE:AMPL {}".format(ch,max(data)*2))
#         self.dev.write("SOURCE{}:VOLTAGE:LOW {}".format(ch,min(data)))
#         self.dev.write("SOURCE{}:VOLTAGE:HIGH {}".format(ch,max(data)))
        self.dev.write("SOURCE{}:VOLTAGE:OFFSET 0".format(ch))
        self.dev.write("SOURCE{}:PHASE 0".format(ch))
        self.dev.write("SOURCE{}:PERIOD {}".format(ch,total_time))
        self.dev.write("SOURCE{}:PHASE:SYNC".format(ch))
        
        #triggered burst
        self.dev.write("SOURCE{}:BURST ON".format(ch))
        self.dev.write("SOURCE{}:BURST:NCYC {}".format(ch,cycles))
        self.dev.write("SOURCE{}:BURST:MODE:TRIG".format(ch))

#===========================================================================
#MOGLabs (1.0 tested)
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

#===========================================================================
# MOGLabs
#===========================================================================         

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

    def freq(self,ch,f):
        if type(f) == str:
            self.dev.cmd('FREQ,{},{}'.format(ch,f))
        else:
            self.dev.cmd('FREQ,{},{} MHz'.format(ch,f))

    def lev(self,ch,f):
        if type(f) == str:
            self.dev.cmd('POW,{},{}'.format(ch,f))
        else:
            self.dev.cmd('POW,{},{} dBm'.format(ch,f))

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
# Oscilloscope DS1102Z_E,  Pete (v2.6 tested)
#===========================================================================            
class Rigol_DS1102Z_E:
    def __init__(self,address): # 'TCPIP::<IP ADDRESS>::INSTR'
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()

    def meas(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    def run(self):
        self.dev.write(':RUN')
    def stop(self):
        self.dev.write(':STOP')
    def single(self):
        self.dev.write(':SINGle')
    
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
# Quantum_Composer (1.0 tested)
#===========================================================================
def sleep_method(method, *args, **kws):
    t_sleep = 50e-3
    def sleeping_method(method, *args, **kws):
        method(*args, **kws)
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
            self.baud_rate = arg
            self.rm = pyvisa.ResourceManager()
            self.dev = self.rm.open_resource(address, # same as 'ASRL5::INSTR'
                                  baud_rate = self.baud_rate, # must identify
                                  data_bits = 8,
                                  parity = Parity.none,
                                  stop_bits = StopBits.one)
            self.mux_reset()   # clear all multiplexer
            self.lev()    # set all outputs to TTL
            self.t_sleep = 50e-3
        else:
            self.address = address
            self.rm = pyvisa.ResourceManager()
            self.dev = self.rm.open_resource(address)
            self.mux_reset()   # clear all multiplexer
            self.lev()    # set all outputs to TTL
            self.dev.clear()
            self.t_sleep = 50e-3

    @sleep_method
    def t0(self,t):  # clock T0
        self.dev.write(':PULSE0:PER {}'.format(t))
    @sleep_method
    def norm(self,*ch):   # normal mode, no wait
        if bool(ch) == True:  # specified channel
            for __ch in ch:
                self.dev.write(':Pulse{}:CMODe NORMal'.format(__ch))
                self.wcount(__ch,0)
        else:
            for __ch in range(1,9):  # all channels
                self.dev.write(':Pulse{}:CMODe NORMal'.format(__ch))
                self.wcount(__ch,0)
    @sleep_method            
    def wid(self,ch,w):
        self.dev.write(':PULSE{}:WIDth {}'.format(ch,w))
    @sleep_method
    def dly(self,ch,d):
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
        elif re.search('c.*(eit)', cfg,re.IGNORECASE) != None:
            self.mux_reset()
            self.off(1,2,3,4,5,6,7,8,0)
            self.trigOff() ## setup system
            __T0 = 100e-6
            __tExp = 3.3
            __tResp = 0e-6  # time that cooling may remain ~0.2 us
            self.dev.write(':PULSE0:PER {}'.format(__T0))  # 0.000100080   
            ## Channel A: Cooling & Magnetic Field 
            self.__exp(__T0,1,'NORM',__tExp,0.3,0.0,1,30000) 
            ## Channel B: Repumper
            self.__exp(__T0,2,'INV',__tExp,0.3,0.0,1,30000)
            ## Channel C: Probe
            self.__exp(__T0,3,'NORM',__tExp,0.1,__tResp,1,30000)
            ## Channel D: Control
            self.__exp(__T0,4,'NORM',__tExp,0.000050,__tResp,1000,30000)
            ## Channel H: TTL for quTAU
            self.__exp(__T0,8,'NORM',__tExp,0.000050,__tResp,1000,30000)
            self.on(1,2,3,4,8)   # this won't start the trigger mode
        ## DLCZ_1, d2 photon generation
        elif (re.search('(photon)|(d2)', cfg ,re.IGNORECASE) != None) and (re.search('(no wait)', cfg ,re.IGNORECASE) == None):  
            self.mux_reset()
            self.off(1,2,3,4,5,6,7,8,0)
            self.trigOff() ## setup system
#            __T0 = 100e-6
            __T0 = 10e-6
            __tExp = 3.3
            self.dev.write(':PULSE0:PER {}'.format(__T0))  # 0.000100080   
            ## Channel A: Cooling & Magnetic Field 
            self.__exp(__T0,1,'INV',__tExp,3.0,0.0,1,0) 
            ## Channel B: Repumper
            self.__exp(__T0,2,'NORM',__tExp,3.0,0.0,1,0)
            ## Channel D: Pump
            self.__exp(__T0,4,'NORM',__tExp,0.000005,0.0,10000,300001)
#             self.__exp(__T0,4,'NORM',__tExp,0.0000025,0.0,10000,300001)
            ## Channel H: TTL for quTAU
            self.__exp(__T0,8,'NORM',__tExp,0.000005,0.0,10000,300000) 
#             self.__exp(__T0,8,'NORM',__tExp,0.0000025,0.0,10000,300000)
            self.on(1,2,4,8)
        elif re.search('(d2 photon no wait)',cfg,re.IGNORECASE) != None:  
            self.mux_reset()
            self.off(1,2,3,4,5,6,7,8,0)
            self.trigOff() ## setup system
            __T0 = 100e-6
            __tExp = 3.3
            __tResp = 0e-6  # no waiting 
            self.dev.write(':PULSE0:PER {}'.format(__T0))  # 0.000100080   
            ## Channel A: Cooling & Magnetic Field 
            self.__exp(__T0,1,'INV',__tExp,3.0,0.0,1,0) 
            ## Channel B: Repumper
            self.__exp(__T0,2,'NORM',__tExp,3.0,0.0,1,0)
            ## Channel D: Pump
            self.__exp(__T0,4,'NORM',__tExp,0.000050,__tResp,1000,30000)
            ## Channel H: TTL for quTAU
            self.__exp(__T0,8,'NORM',__tExp,0.000050,__tResp,1000,30000)
            self.on(1,2,4,8)
        elif re.search('(fluo)',cfg,re.IGNORECASE) != None:    # fluorescence
            self.mux_reset()
            self.off(1,2,3,4,5,6,7,8,0)
            self.trigOff() ## setup system
            __T0 = 100e-6
            __tExp = 3.3
            __tResp = 0e-6  # no waiting 
            self.dev.write(':PULSE0:PER {}'.format(__T0))  # 0.000100080   
            ## Channel A: Magnetic Field 
            self.__exp(__T0,1,'INV',__tExp,3.0,0.0,1,0) 
            ## Channel B: Repumper
            if re.search('(repump)',cfg,re.IGNORECASE) != None:   # leave repump on
                __repumpTime = 3.0 + 0.1
            else:
                __repumpTime = 3.0
            self.__exp(__T0,2,'NORM',__tExp,__repumpTime,0.0,1,0)
            ## Channel E: Cooling
            if re.search('(cool)',cfg,re.IGNORECASE) != None:   # leave cooling on
                __coolTime = 3.0 + 0.1
            else:
                __coolTime = 3.0
            self.__exp(__T0,5,'INV',__tExp,__coolTime,0.0,1,0) 
            ## added for cooling
            self.__exp(__T0,6,'INV',__tExp,0.1,0.0,1,30001) 
            self.mux(5,5,6)
            ## Channel H: TTL for quTAU
            self.__exp(__T0,8,'NORM',__tExp,0.000050,__tResp,1000,30000)
            self.on(1,2,5,6,8)
    @sleep_method
    def burst(ch, n_pulses):
        self.lev(self,ch)
        self.pol(self,ch,'NORM')
        self.norm(self,ch)
        self.on(self,ch)
        self.trigOff(self)
        self.dev.write(':PULSe{}:CMOD BURS'.format(ch))
        time.sleep(self.t_sleep)
        self.dev.write(':PULSe{}:BCOunter {}'.format(ch,n_pulses))
        

#================================================================
# Rigol_DSA832 Spectrum Analyzer ()
#================================================================
class Rigol_DSA832:
    def __init__(self,address): #'TCPIP::<IP ADDRESS>::INSTR' 
        self.address = address            
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.dev.write_termination='\n'
        self.dev.read_termination='\n'

    def freq(self,*f):  # center frequency
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:FREQuency:CENTer {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:FREQuency:CENTer?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq

    def span(self,*f):  # frequency range 
        if bool(f) == True: 
            __f = f[0] 
            self.dev.write(':SENSe:FREQuency:SPAN {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:FREQuency:SPAN?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq

    def rbw(self,*f):  # resolution bandwidth
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:BANDwidth:RESolution {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:BANDwidth:RESolution?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq
    def vbw(self,*f):  # video bandwidth
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:BANDwidth:VIDeo {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:BANDwidth:VIDeo?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq
        
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

    def auto(self): # auto VBW and RBW
        self.dev.write(':SENSe:BANDwidth:RESolution:AUTO ON')
        self.dev.write(':SENSe:BANDwidth:VIDeo:AUTO ON')
        
    def set_frequency_bounds(self,f_lower,f_upper):
        self.dev.write('SENS:FREQ:STARt {}'.format(f_lower))
        self.dev.write('SENS:FREQ:STOP {}'.format(f_upper))        
        
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
        
        peaks, properties = find_peaks(data, prominence=1,width=20)
        print('Freq. Peaks = ',freq[peaks],'Hz')
        print('Amp. Peaks = ',data[peaks],'dBm')
        
        ##plot
        
        plt.plot(freq,data)
        plt.plot(freq[peaks],data[peaks],'x',ms=15,label=str(freq[peaks]))
        plt.legend()
        plt.xlabel('Freq [Hz]')
        plt.ylabel('Amp [dBm]')
        
        return 
    
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

    def lev(self,amp):  #define ampitude and unit {dBm,dBUV,V,VEMF}
        #self.dev.write('POWer:AMPLitude {} {}'.format(amp,unit))
        self.dev.write('POWer:AMPLitude {} dBm'.format(amp))
        
    def offset(self,amp,unit):  #define offset and unit {dBm,dBUV,V,VEMF}
        self.dev.write(':POWer:OFFSet {} {}'.format(amp,unit))

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

    def __Hz(self,f):  # in Hz, support unit. Default: MHz
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

    def offset(self,o):  # V_DC
        self.dev.write('SOURC:VOLTage:OFFSet {}'.format(o));

    def phase(self,p):
        self.dev.write('SOURC:PHASe {}'.format(p));
        
    def burst_delay(self,t):
        self.dev.write('SOUR:BURS:TDEL {}ns'.format(t))
        
    def gaussian(self,t,mu,FWHM,a): #Gaussian Function. Inputs: (FWHM, Amplitude, Center)
        sigma = (FWHM)/(2*np.sqrt(2*np.log(2)))
        amplitude = np.sqrt(2*np.pi*sigma**2)*a
        return amplitude*( 1/(sigma * np.sqrt(2*np.pi) ) )*np.exp( -((t-mu)**2 / (2*sigma**2)) )
    
    def square(self,t,leadingedge,width,amp): #square pulse with duty cycle
        return np.piecewise(t,[(t<=leadingedge),((t>leadingedge) & (t<leadingedge+width)),(t>=leadingedge+width)],[0,amp,0])
    
    def arb(self,freq,func,*arg):
        total_time = 1/(self.__Hz(freq))
        
        if inspect.ismethod(func) == True:
            
            t = np.linspace(0,total_time,2**13)
            data = func(t,*arg)
            
            datastring = self.normalize(data)
            m = 16382 / (datastring.max() - datastring.min())
            b = -m * datastring.min()
            dac_values = (m * datastring + b)
            np.around(dac_values, out=dac_values)
            dac_values = dac_values.astype(np.uint16)
            
        else:
            data = func
            datastring = self.normalize(data)
            m = 16382 / (datastring.max() - datastring.min())
            b = -m * datastring.min()
            dac_values = (m * datastring + b)
            np.around(dac_values, out=dac_values)
            dac_values = dac_values.astype(np.uint16)
        
        self.dev.write("SOURCE:BURST OFF")
        self.dev.write("OUTPUT ON")
        self.dev.write("DATA:DEF EMEM,", str(len(data)))
        self.dev.write_binary_values("DATA:DATA EMEM1,", dac_values, datatype="H", is_big_endian=True)
        self.dev.write("SOURce:FUNC:SHAPE EMEM1")
        self.dev.write("SOURce1:VOLTage:LEVel:IMMediate:LOW {}".format(min(data)))
        self.dev.write("SOURce1:VOLTage:LEVel:IMMediate:HIGH {}".format(max(data)))
        self.dev.write("SOURCE:PHASE 0")
        self.dev.write("SOURCE:PERIOD {}".format(total_time))
    
    def ext_trig(self):
        self.dev.write("TRIGger:SEQuence:SOURce EXTernal")
        
    def normalize(self,data):
        return np.array(data)/np.absolute(max(data))
    
    def arb_burst(self,freq,cycles,func,*arg):
        total_time = 1/(self.__Hz(freq))
        
        if inspect.ismethod(func) == True:
            
            t = np.linspace(0,total_time,2**13)
            data = func(t,*arg)
            
            datastring = self.normalize(data)
            m = 16382 / (datastring.max() - datastring.min())
            b = -m * datastring.min()
            dac_values = (m * datastring + b)
            np.around(dac_values, out=dac_values)
            dac_values = dac_values.astype(np.uint16)
            
        else:
            data = func
            datastring = self.normalize(data)
            m = 16382 / (datastring.max() - datastring.min())
            b = -m * datastring.min()
            dac_values = (m * datastring + b)
            np.around(dac_values, out=dac_values)
            dac_values = dac_values.astype(np.uint16)
        
        self.dev.write("OUTPUT ON")
        self.dev.write("DATA:DEF EMEM,", str(len(data)))
        self.dev.write_binary_values("DATA:DATA EMEM1,", dac_values, datatype="H", is_big_endian=True)
        self.dev.write("SOURce:FUNC:SHAPE EMEM1")
        self.dev.write("SOURce1:VOLTage:LEVel:IMMediate:LOW {}".format(min(data)))
        self.dev.write("SOURce1:VOLTage:LEVel:IMMediate:HIGH {}".format(max(data)))
        self.dev.write("SOURCE:PHASE 0")
        self.dev.write("SOURCE:PERIOD {}".format(total_time))
    
        #triggered burst
        self.dev.write("SOURCE:BURST:STAT ON")
        self.dev.write("SOURCE:BURST:NCYC {}".format(cycles))
        self.dev.write("SOURCE:BURST:MODE TRIG")
        
    def burst(self,number,amp):
        self.dev.write("SOUR:BURS:MODE TRIG")
        self.dev.write("SOUR:BURS:NCYC {}".format(number))
        self.dev.write("SOUR:BURS:MODE TRIG")
        self.dev.write("SOUR:BURS:STAT ON")
        self.dev.write("SOUR:BURS:DEL {}".format(0))
        self.dev.write("SOUR:FUNC:SHAP SQU")
        self.dev.write("SOUR:VOLT:LEV:IMM:AMPL {}V".format(amp))
        self.dev.write("TRIG:SEQ:SOUR EXT")
        self.dev.write("SOURCE:VOLTAGE:LEV:IMM:OFFSET 1V")
        self.dev.write("OUTP:STAT ON")
        
#===========================================================================
# Oscilloscope DS1104
#===========================================================================            
class Rigol_DS1104:
    def __init__(self,address):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()

    def meas(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    
    def run(self):
        self.dev.write(':RUN')
    def stop(self):
        self.dev.write(':STOP')
    def single(self):
        self.dev.write(':SINGle')
    
    def screenshot(self,ch):
        
        timescale = float(self.dev.query(":TIM:SCAL?"))        
        timeoffset = float(self.dev.query(":TIM:OFFS?")[0])# Get the timescale offset
        voltscale = float(self.dev.query(":CHAN{}:SCAL?".format(ch)))
        voltoffset = float(str(self.dev.query(":CHAN{}:OFFS?".format(ch))))  # And the voltage offset
        
        self.dev.write(":WAV:SOUR: CHAN{}".format(ch))
        self.dev.write(":WAV:POIN:MODE NORM")
        self.dev.write(":WAV:DATA? CHAN{}".format(ch))
        rawdata = self.dev.read_raw() #1024 data
        rawdata = rawdata[11:]

        data = np.frombuffer(rawdata, 'B')
        
        # Walk through the data, and map it to actual voltages
        data = data + 254
        data = data[:-1]

        # Now, we know from experimentation that the scope display range is actually
        # 30-229.  So shift by 130 - the voltage offset in counts, then scale to
        # get the actual voltage.
        data = (data - 130.0 - voltoffset/voltscale*25) / 25 * voltscale
        
        return data

#===========================================================================
# Oscilloscope Rigol DMO5000
#=========================================================================== 

class Rigol_DMO5000:
    def __init__(self,address):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()

    def meas(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    def run(self):
        self.dev.write(':RUN')
    def stop(self):
        self.dev.write(':STOP')
    def single(self):
        self.dev.write(':SINGle')
        
    def screenshot(self,ch):
        self.dev.write(":WAV:SOUR: CHAN{}".format(ch))
        self.dev.write(":WAV:MODE NORMal")
        self.dev.write(":WAV:FORM ASC")
        self.dev.write(":WAV:DATA? CHAN{}".format(ch))
        rawdata = self.dev.read_raw()
        rawdata = rawdata.decode('UTF-8')
        volt_data = rawdata[11:-2] #removes header and ending of data
        volt_data = np.array([float(data) for data in data.split(',')])
        
        t = float(self.dev.query(':WAVeform:XINCrement?'))
        time_data = np.arange(0,t*len(volt_data),t)
        
        return time_data,volt_data