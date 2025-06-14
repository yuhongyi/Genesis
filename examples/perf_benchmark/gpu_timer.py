import torch

class GPU_Timer:
    def __init__(self):
        self.status = "idle"
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.elapsed_time_ms = 0

    def start(self):
        if self.status == "idle":
            self.start.record()
            self.elapsed_time_ms = 0
            self.status = "running"
        else:
            raise Exception("GPU timer is already running")

    def end(self):
        if self.status == "running":
            self.end.record()
            self.elapsed_time_ms += self.start.elapsed_time(self.end)
            self.status = "idle"
        else:
            raise Exception("GPU timer is not running")

    def pause(self):
        if self.status == "running":
            self.end.record()
            self.elapsed_time_ms += self.start.elapsed_time(self.end)
            self.status = "paused"
        else:
            raise Exception("GPU timer is not running")
            
    def resume(self):
        if self.status == "paused":
            self.start.record()
            self.status = "running"
        else:
            raise Exception("GPU timer is not paused")
            
    def get_elapsed_time(self):
        if self.status == "running":
            current = torch.cuda.Event(enable_timing=True)
            current.record()
            return self.elapsed_time_ms + self.start.elapsed_time(current)
        else:
            raise Exception("GPU timer is not running")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()  # GPU "fence" / marker

# GPU task you want to profile
output = model(input)

end.record()

# Wait for everything to finish
torch.cuda.synchronize()

# Measure elapsed time in milliseconds
elapsed_time_ms = start.elapsed_time(end)
print(f"Elapsed time: {elapsed_time_ms:.3f} ms")
