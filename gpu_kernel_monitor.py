#!/usr/bin/env python3
"""
GPU Kernel-Level Monitor
Advanced CUDA kernel and operator-level monitoring for NVIDIA GPUs
"""

import time
import subprocess
import re
import json
import threading
import signal
import sys
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Tuple
import xml.etree.ElementTree as ET

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    try:
        import nvidia_ml_py3 as pynvml
        PYNVML_AVAILABLE = True  
    except ImportError:
        PYNVML_AVAILABLE = False
        pynvml = None

import psutil
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.align import Align
from rich.tree import Tree
from rich import box

class KernelInfo:
    """Container for CUDA kernel execution information"""
    def __init__(self):
        self.name: str = ""
        self.pid: int = 0
        self.process_name: str = ""
        self.duration_us: float = 0.0
        self.grid_size: Tuple[int, int, int] = (0, 0, 0)
        self.block_size: Tuple[int, int, int] = (0, 0, 0)
        self.shared_memory: int = 0
        self.registers_per_thread: int = 0
        self.timestamp: datetime = datetime.now()
        self.gpu_id: int = 0
        self.stream_id: int = 0
        self.context_id: int = 0

class MemoryTransfer:
    """Container for GPU memory transfer information"""
    def __init__(self):
        self.transfer_type: str = ""  # H2D, D2H, D2D
        self.size_bytes: int = 0
        self.duration_us: float = 0.0
        self.bandwidth_gbps: float = 0.0
        self.pid: int = 0
        self.process_name: str = ""
        self.timestamp: datetime = datetime.now()
        self.gpu_id: int = 0

class GPUKernelMonitor:
    def __init__(self, history_size=1000):
        self.console = Console()
        self.history_size = history_size
        self.running = True
        
        # Data storage
        self.kernel_history: deque = deque(maxlen=history_size)
        self.memory_transfers: deque = deque(maxlen=history_size)
        self.active_processes: Dict[int, Dict[str, Any]] = {}
        
        # Performance statistics
        self.kernel_stats = {
            'total_kernels': 0,
            'total_duration_ms': 0.0,
            'avg_kernel_duration_us': 0.0,
            'most_active_process': '',
            'kernels_per_second': 0.0,
            'memory_bandwidth_gbps': 0.0
        }
        
        # Process tracking
        self.process_kernel_count: Dict[int, int] = defaultdict(int)
        self.process_memory_usage: Dict[int, int] = defaultdict(int)
        
        # Initialize NVIDIA ML
        self.nvml_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.console.print(f"[green]✅ NVIDIA ML initialized - {self.gpu_count} GPU(s)[/green]")
            except Exception as e:
                self.console.print(f"[red]❌ NVIDIA ML init failed: {e}[/red]")
        
        # Check for profiling tools
        self.profiling_available = self.check_profiling_tools()
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start monitoring thread
        self.monitoring_thread = None
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        self.console.print("\n[yellow]🛑 Shutting down kernel monitor...[/yellow]")
        sys.exit(0)
    
    def check_profiling_tools(self) -> bool:
        """Check availability of NVIDIA profiling tools"""
        tools = ['nvidia-smi', 'nsight-sys', 'ncu']
        available_tools = []
        
        for tool in tools:
            try:
                result = subprocess.run(['which', tool], capture_output=True, text=True)
                if result.returncode == 0:
                    available_tools.append(tool)
            except FileNotFoundError:
                pass
        
        if available_tools:
            self.console.print(f"[green]✅ Profiling tools available: {', '.join(available_tools)}[/green]")
            return True
        else:
            self.console.print("[yellow]⚠️ Limited profiling - install nsight-systems for detailed kernel tracking[/yellow]")
            return False
    
    def get_gpu_processes_detailed(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed information about GPU processes"""
        processes = {}
        
        if not self.nvml_available:
            return processes
        
        for gpu_id in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                for proc in gpu_processes:
                    pid = proc.pid
                    if pid not in processes:
                        processes[pid] = {
                            'pid': pid,
                            'name': 'Unknown',
                            'command': 'Unknown',
                            'gpu_memory_usage': {},
                            'total_gpu_memory': 0,
                            'cpu_percent': 0.0,
                            'memory_percent': 0.0,
                            'create_time': 0,
                            'status': 'unknown'
                        }
                    
                    processes[pid]['gpu_memory_usage'][gpu_id] = proc.usedGpuMemory
                    processes[pid]['total_gpu_memory'] += proc.usedGpuMemory
                    
                    # Get process details
                    try:
                        ps_proc = psutil.Process(pid)
                        processes[pid]['name'] = ps_proc.name()
                        processes[pid]['command'] = ' '.join(ps_proc.cmdline()[:5])
                        processes[pid]['cpu_percent'] = ps_proc.cpu_percent()
                        processes[pid]['memory_percent'] = ps_proc.memory_percent()
                        processes[pid]['create_time'] = ps_proc.create_time()
                        processes[pid]['status'] = ps_proc.status()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
            except Exception as e:
                self.console.print(f"[red]Error getting GPU {gpu_id} processes: {e}[/red]")
        
        return processes
    
    def monitor_nvidia_smi_processes(self):
        """Monitor GPU processes using nvidia-smi for process-level insights"""
        try:
            # Use nvidia-smi to get process information with more details
            cmd = ['nvidia-smi', 'pmon', '-c', '1', '-s', 'um']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    # Parse nvidia-smi pmon output
                    # Format: gpu pid type sm mem enc dec command
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            gpu_id = int(parts[0])
                            pid = int(parts[1])
                            proc_type = parts[2]  # C = Compute, G = Graphics
                            sm_util = parts[3]    # SM utilization
                            mem_util = parts[4]   # Memory utilization  
                            enc_util = parts[5]   # Encoder utilization
                            dec_util = parts[6]   # Decoder utilization
                            
                            # Store process activity
                            if pid not in self.active_processes:
                                self.active_processes[pid] = {
                                    'name': 'Unknown',
                                    'type': proc_type,
                                    'gpu_activity': {},
                                    'last_seen': datetime.now()
                                }
                            
                            self.active_processes[pid]['gpu_activity'][gpu_id] = {
                                'sm_util': sm_util,
                                'mem_util': mem_util,
                                'enc_util': enc_util,
                                'dec_util': dec_util,
                                'timestamp': datetime.now()
                            }
                            self.active_processes[pid]['last_seen'] = datetime.now()
                            
                        except (ValueError, IndexError):
                            continue
                            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pass  # nvidia-smi not available or timeout
    
    def parse_nsight_output(self, output: str) -> List[KernelInfo]:
        """Parse nsight-systems output to extract kernel information"""
        kernels = []
        
        # This would parse actual nsight-sys output
        # For now, we'll simulate kernel detection based on process activity
        return kernels
    
    def estimate_kernel_activity(self) -> List[Dict[str, Any]]:
        """Estimate kernel activity based on GPU utilization patterns"""
        estimated_kernels = []
        
        if not self.nvml_available:
            return estimated_kernels
        
        for gpu_id in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Get current utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util_info.gpu
                mem_util = util_info.memory
                
                # Get memory info for transfer estimation
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get processes
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                for proc in processes:
                    if gpu_util > 10:  # Only if GPU is active
                        # Estimate kernel execution based on utilization
                        estimated_kernel = {
                            'gpu_id': gpu_id,
                            'pid': proc.pid,
                            'estimated_duration_us': gpu_util * 1000,  # Rough estimate
                            'memory_usage_mb': proc.usedGpuMemory / 1024**2,
                            'timestamp': datetime.now(),
                            'utilization': gpu_util,
                            'memory_utilization': mem_util
                        }
                        estimated_kernels.append(estimated_kernel)
                        
            except Exception as e:
                continue
        
        return estimated_kernels
    
    def create_kernel_activity_panel(self) -> Panel:
        """Create panel showing kernel activity"""
        if not self.active_processes:
            return Panel(
                Align.center(Text("No active GPU processes detected", style="yellow")),
                title="🔥 Kernel Activity",
                border_style="yellow"
            )
        
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("PID", style="cyan", width=8)
        table.add_column("Process", style="green", width=20)
        table.add_column("Type", style="yellow", width=6)
        table.add_column("GPU", style="blue", width=6)
        table.add_column("SM%", style="red", width=6)
        table.add_column("MEM%", style="magenta", width=6)
        table.add_column("Activity", style="white", width=20)
        
        for pid, proc_info in self.active_processes.items():
            if datetime.now() - proc_info['last_seen'] > timedelta(seconds=10):
                continue  # Skip stale processes
                
            try:
                ps_proc = psutil.Process(pid)
                proc_name = ps_proc.name()[:18] + ".." if len(ps_proc.name()) > 20 else ps_proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc_name = "Unknown"
            
            for gpu_id, activity in proc_info['gpu_activity'].items():
                # Create activity indicator
                sm_val = activity['sm_util']
                mem_val = activity['mem_util']
                
                if sm_val != '-' and mem_val != '-':
                    try:
                        sm_int = int(sm_val)
                        mem_int = int(mem_val)
                        
                        # Activity level
                        if sm_int > 80 or mem_int > 80:
                            activity_text = "🔥 HIGH"
                            activity_color = "red"
                        elif sm_int > 50 or mem_int > 50:
                            activity_text = "⚡ MEDIUM" 
                            activity_color = "yellow"
                        elif sm_int > 10 or mem_int > 10:
                            activity_text = "💫 LOW"
                            activity_color = "green"
                        else:
                            activity_text = "💤 IDLE"
                            activity_color = "dim"
                    except ValueError:
                        sm_int, mem_int = 0, 0
                        activity_text = "❓ UNKNOWN"
                        activity_color = "dim"
                else:
                    sm_int, mem_int = 0, 0
                    activity_text = "💤 IDLE"
                    activity_color = "dim"
                
                table.add_row(
                    str(pid),
                    proc_name,
                    proc_info['type'],
                    f"GPU{gpu_id}",
                    f"{sm_val}%",
                    f"{mem_val}%",
                    f"[{activity_color}]{activity_text}[/{activity_color}]"
                )
        
        return Panel(table, title="🔥 GPU Kernel Activity (Estimated)", border_style="red")
    
    def create_memory_operations_panel(self) -> Panel:
        """Create panel showing memory operations"""
        if not self.nvml_available:
            return Panel(
                Align.center(Text("NVIDIA ML not available", style="red")),
                title="💾 Memory Operations",
                border_style="red"
            )
        
        table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
        table.add_column("GPU", style="cyan", width=6)
        table.add_column("Used", style="yellow", width=12)
        table.add_column("Total", style="blue", width=12)
        table.add_column("Bandwidth", style="green", width=12)
        table.add_column("Activity", style="white", width=15)
        
        total_bandwidth = 0
        
        for gpu_id in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_gb = mem_info.used / 1024**3
                mem_total_gb = mem_info.total / 1024**3
                mem_percent = (mem_info.used / mem_info.total) * 100
                
                # Memory utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_util = util_info.memory
                
                # Estimate bandwidth based on memory utilization
                # This is approximate - real bandwidth requires profiling tools
                estimated_bandwidth = mem_util * 10  # GB/s estimate based on utilization
                total_bandwidth += estimated_bandwidth
                
                # Activity level
                if mem_percent > 90:
                    activity = "🔴 FULL"
                    activity_color = "red"
                elif mem_percent > 70:
                    activity = "🟡 HIGH"
                    activity_color = "yellow"
                elif mem_percent > 30:
                    activity = "🟢 ACTIVE"
                    activity_color = "green"
                else:
                    activity = "⚪ LOW"
                    activity_color = "dim"
                
                table.add_row(
                    f"GPU{gpu_id}",
                    f"{mem_used_gb:.1f} GB",
                    f"{mem_total_gb:.1f} GB",
                    f"{estimated_bandwidth:.1f} GB/s",
                    f"[{activity_color}]{activity}[/{activity_color}]"
                )
                
            except Exception:
                table.add_row(f"GPU{gpu_id}", "Error", "Error", "Error", "[red]ERROR[/red]")
        
        # Update stats
        self.kernel_stats['memory_bandwidth_gbps'] = total_bandwidth
        
        return Panel(table, title="💾 Memory Operations", border_style="blue")
    
    def create_process_breakdown_panel(self) -> Panel:
        """Create detailed process breakdown panel"""
        processes = self.get_gpu_processes_detailed()
        
        if not processes:
            return Panel(
                Align.center(Text("No GPU processes detected", style="yellow")),
                title="📊 Process Breakdown",
                border_style="yellow"
            )
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("PID", style="cyan", width=8)
        table.add_column("Process", style="green", width=25)
        table.add_column("GPU Mem", style="yellow", width=12)
        table.add_column("CPU%", style="red", width=8)
        table.add_column("Runtime", style="blue", width=10)
        table.add_column("Status", style="white", width=12)
        
        # Sort by GPU memory usage
        sorted_processes = sorted(processes.values(), 
                                key=lambda x: x['total_gpu_memory'], reverse=True)
        
        for proc in sorted_processes[:15]:  # Top 15 processes
            # Calculate runtime
            if proc['create_time'] > 0:
                runtime = datetime.now().timestamp() - proc['create_time']
                if runtime > 3600:
                    runtime_str = f"{runtime/3600:.1f}h"
                elif runtime > 60:
                    runtime_str = f"{runtime/60:.1f}m"
                else:
                    runtime_str = f"{runtime:.0f}s"
            else:
                runtime_str = "Unknown"
            
            # Status with color
            status = proc['status'].upper()
            if status == 'RUNNING':
                status_text = f"[green]{status}[/green]"
            elif status in ['SLEEPING', 'WAITING']:
                status_text = f"[yellow]{status}[/yellow]"
            else:
                status_text = f"[dim]{status}[/dim]"
            
            table.add_row(
                str(proc['pid']),
                proc['name'][:23] + ".." if len(proc['name']) > 25 else proc['name'],
                f"{proc['total_gpu_memory'] / 1024**2:.0f} MB",
                f"{proc['cpu_percent']:.1f}%",
                runtime_str,
                status_text
            )
        
        return Panel(table, title="📊 GPU Process Breakdown", border_style="magenta")
    
    def create_performance_stats_panel(self) -> Panel:
        """Create performance statistics panel"""
        # Calculate kernels per second (estimated)
        active_processes_count = len([p for p in self.active_processes.values() 
                                    if datetime.now() - p['last_seen'] < timedelta(seconds=5)])
        
        # Estimate kernel activity
        estimated_kernels = self.estimate_kernel_activity()
        current_kernel_activity = len(estimated_kernels)
        
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        table.add_column("Indicator", style="white", width=10)
        
        table.add_row("Active Processes", str(active_processes_count), "🔄" if active_processes_count > 0 else "💤")
        table.add_row("Kernel Activity", str(current_kernel_activity), "🔥" if current_kernel_activity > 0 else "❄️")
        table.add_row("Memory Bandwidth", f"{self.kernel_stats['memory_bandwidth_gbps']:.1f} GB/s", "📊")
        
        # GPU utilization summary
        total_util = 0
        if self.nvml_available:
            for gpu_id in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    total_util += util_info.gpu
                except Exception:
                    pass
        
        avg_util = total_util / self.gpu_count if self.gpu_count > 0 else 0
        table.add_row("Avg GPU Util", f"{avg_util:.1f}%", "⚡" if avg_util > 50 else "🐌")
        
        # Profiling status
        profiling_status = "✅ Available" if self.profiling_available else "❌ Limited"
        table.add_row("Profiling Tools", profiling_status, "🔍")
        
        return Panel(table, title="📈 Performance Statistics", border_style="green")
    
    def start_monitoring_thread(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while self.running:
                try:
                    self.monitor_nvidia_smi_processes()
                    time.sleep(2)
                except Exception as e:
                    self.console.print(f"[red]Monitoring error: {e}[/red]")
                    time.sleep(5)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def run_kernel_monitor(self):
        """Run the kernel-level monitoring system"""
        self.console.clear()
        self.console.print("[bold green]🚀 Starting GPU Kernel Monitor...[/bold green]")
        
        if not self.nvml_available:
            self.console.print("[red]❌ NVIDIA ML not available. Limited functionality.[/red]")
        
        time.sleep(2)
        
        # Start background monitoring
        self.start_monitoring_thread()
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=4)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="kernel_activity"),
            Layout(name="memory_ops", size=12)
        )
        
        layout["right"].split_column(
            Layout(name="process_breakdown"),
            Layout(name="stats", size=12)
        )
        
        with Live(layout, console=self.console, screen=True, auto_refresh=False) as live:
            while self.running:
                try:
                    # Update header
                    layout["header"].update(
                        Panel(
                            Text("🔥 GPU Kernel-Level Monitor - CUDA Operations & Memory Tracking", 
                                 style="bold white", justify="center"),
                            style="bold red"
                        )
                    )
                    
                    # Update panels
                    layout["kernel_activity"].update(self.create_kernel_activity_panel())
                    layout["memory_ops"].update(self.create_memory_operations_panel())
                    layout["process_breakdown"].update(self.create_process_breakdown_panel())
                    layout["stats"].update(self.create_performance_stats_panel())
                    
                    # Update footer
                    layout["footer"].update(
                        Panel(
                            Text("Press Ctrl+C to exit | Install nsight-systems for detailed kernel profiling | Updates every 2s", 
                                 style="dim white", justify="center"),
                            style="dim blue"
                        )
                    )
                    
                    live.refresh()
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Display error: {e}[/red]")
                    time.sleep(2)

def main():
    """Main entry point"""
    monitor = GPUKernelMonitor()
    monitor.run_kernel_monitor()

if __name__ == "__main__":
    main()