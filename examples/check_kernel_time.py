import time # wall time
import resource # kernel time
from tqdm.auto import tqdm

start_time = time.process_time()
start_resource = resource.getrusage(resource.RUSAGE_SELF)

sum = 0
for i in tqdm(range(500_000_000)):
    sum += i
    
end_time = time.process_time()
end_resource = resource.getrusage(resource.RUSAGE_SELF)

print("wall time: ", end_time - start_time, "s")
print("kernel time: ", end_resource.ru_stime - start_resource.ru_stime, "s")