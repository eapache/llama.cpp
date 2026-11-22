import ctypes as c
import threading

lib = c.CDLL("libnvidia-ml.so.1")
class Info(c.Structure):
    _fields_ = [("pid", c.c_uint), ("used", c.c_ulonglong), ("gpu", c.c_uint), ("compute", c.c_uint)]

def check(rc):
    if rc:
        raise RuntimeError(f"NVML error {rc}")

def sample(pid):
    check(lib.nvmlInit_v2())
    handle = c.c_void_p()
    check(lib.nvmlDeviceGetHandleByIndex_v2(0,c.byref(handle)))
    def read():
        count = c.c_uint(128)
        items = (Info * count.value)()
        check(lib.nvmlDeviceGetComputeRunningProcesses_v3(handle,c.byref(count),items))
        return next(i.used for i in items[:count.value] if i.pid==pid)
    values = [read()]
    stop = threading.Event()
    def collect():
        while not stop.wait(.1):
            values.append(read())
    thread = threading.Thread(target=collect,daemon=True)
    thread.start()
    def finish():
        stop.set()
        thread.join()
        values.append(read())
        check(lib.nvmlShutdown())
        return dict(ready=values[0], end=values[-1], peak=max(values), samples=len(values))
    return finish
