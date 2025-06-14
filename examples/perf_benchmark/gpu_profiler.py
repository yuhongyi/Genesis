import torch
import numpy as np

class GPUProfiler:
    def __init__(self, n_steps):
        self.reset(n_steps)

    def reset(self, n_steps):
        self.n_steps = n_steps
        # Create arrays of CUDA events for each step
        # Each step has 3 events: simulation_start, render_start, render_end
        self.events = []
        for _ in range(n_steps):
            step_events = {
                'simulation_start': torch.cuda.Event(enable_timing=True),
                'render_start': torch.cuda.Event(enable_timing=True),
                'render_end': torch.cuda.Event(enable_timing=True)
            }
            self.events.append(step_events)
        self.current_step = 0
        self.is_synchronized = False

    def on_simulation_start(self):
        """Record the start of simulation for current step"""
        if self.current_step >= self.n_steps:
            raise Exception("All steps have been profiled")
        self.events[self.current_step]['simulation_start'].record()

    def on_rendering_start(self):
        """Record the start of rendering for current step"""
        if self.current_step >= self.n_steps:
            raise Exception("All steps have been profiled")
        self.events[self.current_step]['render_start'].record()

    def on_rendering_end(self):
        """Record the end of rendering for current step"""
        if self.current_step >= self.n_steps:
            raise Exception("All steps have been profiled")
        self.events[self.current_step]['render_end'].record()
        self.current_step += 1

    def get_total_simulation_gpu_time_ms(self):
        """Calculate total simulation GPU time across all steps in milliseconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            events = self.events[step]
            total_time += events['simulation_start'].elapsed_time(events['render_start'])
        return total_time

    def get_total_rendering_gpu_time_ms(self):
        """Calculate total rendering GPU time across all steps in milliseconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            events = self.events[step]
            total_time += events['render_start'].elapsed_time(events['render_end'])
        return total_time
    
    def get_total_gpu_time_ms(self):
        """Calculate total GPU time across all steps in milliseconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            events = self.events[step]
            total_time += events['simulation_start'].elapsed_time(events['render_end'])
        return total_time
    
    def end(self):
        """End the profiler"""
        self._synchronize()

    def _synchronize(self):
        """Synchronize GPU to ensure all events are recorded"""
        torch.cuda.synchronize()
        self.is_synchronized = True
