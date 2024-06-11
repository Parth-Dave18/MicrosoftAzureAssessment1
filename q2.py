import time

def log_execution_time(func):
    def wrapper(duration):
        start = time.time()
        execution = func(duration)
        end = time.time()
        print("total execution time = ",end - start)

    return wrapper

@log_execution_time
def delay(duration):
    time.sleep(duration)

delay(2)


# OUTPUT
# total execution time =  2.0114591121673584