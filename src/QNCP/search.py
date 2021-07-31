def get_resource(inputtype,devicetype,*arg):
    dev_list = []
    for i in range(1,20):
        try:
            ip = inputtype+'{}::INSTR'.format(i)
            device = devicetype(ip,*arg)
            dev_list.append(ip)
        except:
            pass
        
    return dev_list
        
def identify(inputtype,devicetype,*arg):
    dev_list = get_resource(inputtype,devicetype,*arg)
    identity = 0
    id_list = []
    for i in range (0,len(dev_list)):
        while identity == 0:
            try:
                device = devicetype(dev_list[i],*arg)
                identity = device.dev.query('*IDN?')
            except:
                pass
        id_list.append(identity)
    return id_list