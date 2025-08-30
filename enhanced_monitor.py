#!/usr/bin/env python3
"""
Enhanced System Monitor with GPU and CPU monitoring
Provides colorful, real-time monitoring with rich console output
"""

import time
import psutil
import GPUtil
try:
    import pynvml
except ImportError:
    try:
        import nvidia_ml_py3 as pynvml
    except ImportError:
        pynvml = None
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.text import Text
from rich import box
from datetime import datetime
import threading
import signal
import sys

class EnhancedSystemMonitor:
    def __init__(self):
        self.console = Console()
        self.running = True
        self.cpu_history = []
        self.gpu_history = []
        self.memory_history = []
        
        # Initialize NVIDIA ML
        try:
            if pynvml:
                pynvml.nvmlInit()
                self.nvml_available = True
            else:
                self.nvml_available = False
        except Exception:
            self.nvml_available = False
            
        # Setup signal handler for graceful exit
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        self.running = False
        self.console.print("\n[yellow]Shutting down monitor...[/yellow]")
        sys.exit(0)
    
    def get_cpu_info(self):
        """Get detailed CPU information"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        return {
            'percent': cpu_percent,
            'avg_percent': sum(cpu_percent) / len(cpu_percent),
            'frequency': cpu_freq.current if cpu_freq else 0,
            'cores': cpu_count,
            'load_avg': load_avg,
            'temp': self.get_cpu_temperature()
        }
    
    def get_cpu_temperature(self):
        """Get CPU temperature if available"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
        except (AttributeError, KeyError, IndexError):
            pass
        return None
    
    def get_memory_info(self):
        """Get memory usage information"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent
        }
    
    def get_gpu_info(self):
        """Get detailed GPU information"""
        gpus = []
        
        try:
            # Using GPUtil for basic info
            gpu_list = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpu_list):
                gpu_info = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature,
                    'power_draw': 0,  # Default
                    'power_limit': 0   # Default
                }
                
                # Enhanced info with pynvml if available
                if self.nvml_available:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        power_info = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                        power_limit = pynvml.nvmlDeviceGetPowerManagementLimitDefault(handle) / 1000
                        
                        gpu_info['power_draw'] = power_info
                        gpu_info['power_limit'] = power_limit
                    except Exception:
                        pass
                
                gpus.append(gpu_info)
                
        except Exception as e:
            self.console.print(f"[red]Error getting GPU info: {e}[/red]")
            
        return gpus
    
    def get_disk_info(self):
        """Get disk usage information"""
        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': (usage.used / usage.total) * 100
                })
            except PermissionError:
                continue
        return disk_info
    
    def get_network_info(self):
        """Get network statistics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
    
    def create_cpu_panel(self, cpu_info):
        """Create CPU monitoring panel"""
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", justify="center")
        
        # Overall CPU usage with color coding
        cpu_color = "red" if cpu_info['avg_percent'] > 80 else "yellow" if cpu_info['avg_percent'] > 60 else "green"
        status = "🔥 HIGH" if cpu_info['avg_percent'] > 80 else "⚠️ MED" if cpu_info['avg_percent'] > 60 else "✅ OK"
        
        table.add_row(
            "CPU Usage", 
            f"{cpu_info['avg_percent']:.1f}%",
            f"[{cpu_color}]{status}[/{cpu_color}]"
        )
        
        table.add_row("CPU Cores", f"{cpu_info['cores']}", "")
        table.add_row("Frequency", f"{cpu_info['frequency']:.0f} MHz", "")
        
        if cpu_info['temp']:
            temp_color = "red" if cpu_info['temp'] > 80 else "yellow" if cpu_info['temp'] > 70 else "green"
            table.add_row("Temperature", f"{cpu_info['temp']:.1f}°C", f"[{temp_color}]{'🔥' if cpu_info['temp'] > 80 else '🌡️'}[/{temp_color}]")
        
        # Load average
        table.add_row("Load Avg (1m)", f"{cpu_info['load_avg'][0]:.2f}", "")
        
        # Individual core usage
        core_usage = Text()
        for i, percent in enumerate(cpu_info['percent']):
            color = "red" if percent > 80 else "yellow" if percent > 60 else "green"
            core_usage.append(f"C{i}: {percent:4.1f}% ", style=color)
            if (i + 1) % 4 == 0:  # New line every 4 cores
                core_usage.append("\n")
        
        return Panel(table, title="🖥️  CPU Monitor", border_style="blue")
    
    def create_memory_panel(self, memory_info):
        """Create memory monitoring panel"""
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Type", style="cyan")
        table.add_column("Used", style="yellow")
        table.add_column("Total", style="blue")
        table.add_column("Percent", justify="center")
        table.add_column("Status", justify="center")
        
        # RAM
        ram_color = "red" if memory_info['percent'] > 85 else "yellow" if memory_info['percent'] > 70 else "green"
        ram_status = "🔥 FULL" if memory_info['percent'] > 85 else "⚠️ HIGH" if memory_info['percent'] > 70 else "✅ OK"
        
        table.add_row(
            "RAM",
            f"{memory_info['used'] / (1024**3):.1f} GB",
            f"{memory_info['total'] / (1024**3):.1f} GB",
            f"[{ram_color}]{memory_info['percent']:.1f}%[/{ram_color}]",
            f"[{ram_color}]{ram_status}[/{ram_color}]"
        )
        
        # Swap
        if memory_info['swap_total'] > 0:
            swap_color = "red" if memory_info['swap_percent'] > 50 else "yellow" if memory_info['swap_percent'] > 25 else "green"
            swap_status = "🔥 HIGH" if memory_info['swap_percent'] > 50 else "⚠️ MED" if memory_info['swap_percent'] > 25 else "✅ OK"
            
            table.add_row(
                "Swap",
                f"{memory_info['swap_used'] / (1024**3):.1f} GB",
                f"{memory_info['swap_total'] / (1024**3):.1f} GB",
                f"[{swap_color}]{memory_info['swap_percent']:.1f}%[/{swap_color}]",
                f"[{swap_color}]{swap_status}[/{swap_color}]"
            )
        
        return Panel(table, title="🧠 Memory Monitor", border_style="green")
    
    def create_gpu_panel(self, gpu_info):
        """Create GPU monitoring panel"""
        if not gpu_info:
            return Panel(
                Text("No GPUs detected or NVIDIA drivers not available", style="red"),
                title="🎮 GPU Monitor", 
                border_style="red"
            )
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("GPU", style="cyan")
        table.add_column("Usage", justify="center")
        table.add_column("Memory", justify="center")
        table.add_column("Temp", justify="center")
        table.add_column("Power", justify="center")
        table.add_column("Status", justify="center")
        
        for gpu in gpu_info:
            # GPU usage color coding
            usage_color = "red" if gpu['load'] > 80 else "yellow" if gpu['load'] > 60 else "green"
            
            # Memory color coding
            mem_color = "red" if gpu['memory_percent'] > 85 else "yellow" if gpu['memory_percent'] > 70 else "green"
            
            # Temperature color coding
            temp_color = "red" if gpu['temperature'] > 80 else "yellow" if gpu['temperature'] > 70 else "green"
            
            # Overall status
            if gpu['load'] > 80 or gpu['memory_percent'] > 85 or gpu['temperature'] > 80:
                status = "🔥 HOT"
                status_color = "red"
            elif gpu['load'] > 60 or gpu['memory_percent'] > 70:
                status = "⚠️ BUSY"
                status_color = "yellow"
            else:
                status = "✅ OK"
                status_color = "green"
            
            power_text = f"{gpu['power_draw']:.0f}W" if gpu['power_draw'] > 0 else "N/A"
            
            table.add_row(
                f"{gpu['name']}",
                f"[{usage_color}]{gpu['load']:.1f}%[/{usage_color}]",
                f"[{mem_color}]{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f}MB ({gpu['memory_percent']:.1f}%)[/{mem_color}]",
                f"[{temp_color}]{gpu['temperature']:.0f}°C[/{temp_color}]",
                power_text,
                f"[{status_color}]{status}[/{status_color}]"
            )
        
        return Panel(table, title="🎮 GPU Monitor", border_style="magenta")
    
    def create_system_panel(self):
        """Create system overview panel"""
        # Get system uptime
        boot_time = psutil.boot_time()
        uptime = datetime.now().timestamp() - boot_time
        uptime_hours = int(uptime // 3600)
        uptime_minutes = int((uptime % 3600) // 60)
        
        # Get process count
        process_count = len(psutil.pids())
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("System Info", style="cyan")
        table.add_column("Value", style="green")
        
        uname_info = psutil.os.uname()
        table.add_row("Hostname", uname_info.nodename)
        table.add_row("OS", f"{uname_info.sysname} {uname_info.release}")
        table.add_row("Uptime", f"{uptime_hours}h {uptime_minutes}m")
        table.add_row("Processes", str(process_count))
        table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return Panel(table, title="🖥️  System Info", border_style="cyan")
    
    def create_layout(self):
        """Create the main dashboard layout"""
        layout = Layout()
        
        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(
            Panel(
                Text("🚀 Enhanced System Monitor - Real-time GPU & CPU Monitoring", 
                     style="bold white", justify="center"),
                style="bold blue"
            )
        )
        
        # Body split into left and right
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="gpu")
        )
        
        # Left side: CPU, Memory, and System
        layout["left"].split_column(
            Layout(name="cpu"),
            Layout(name="memory"),
            Layout(name="system")
        )
        
        # Footer
        layout["footer"].update(
            Panel(
                Text("Press Ctrl+C to exit | Updates every 2 seconds", 
                     style="dim white", justify="center"),
                style="dim blue"
            )
        )
        
        return layout
    
    def run(self):
        """Run the enhanced monitoring dashboard"""
        self.console.clear()
        self.console.print("[bold green]Starting Enhanced System Monitor...[/bold green]")
        time.sleep(1)
        
        layout = self.create_layout()
        
        with Live(layout, console=self.console, screen=True, auto_refresh=False) as live:
            while self.running:
                try:
                    # Gather system information
                    cpu_info = self.get_cpu_info()
                    memory_info = self.get_memory_info()
                    gpu_info = self.get_gpu_info()
                    
                    # Update panels
                    layout["cpu"].update(self.create_cpu_panel(cpu_info))
                    layout["memory"].update(self.create_memory_panel(memory_info))
                    layout["gpu"].update(self.create_gpu_panel(gpu_info))
                    layout["system"].update(self.create_system_panel())
                    
                    # Refresh the display
                    live.refresh()
                    
                    # Wait before next update
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(2)

def main():
    """Main entry point"""
    monitor = EnhancedSystemMonitor()
    monitor.run()

if __name__ == "__main__":
    main()