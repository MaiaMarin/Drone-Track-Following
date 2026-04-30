is_interrupted = False
def checkInterrupt():
    global is_interrupted

    if is_interrupted:
        is_interrupted = False
        raise KeyboardInterrupt("Program interrupted by user.")
    
def triggerInterrupt():
    global is_interrupted
    is_interrupted = True

def clearInterrupt():
    global is_interrupted
    is_interrupted = False