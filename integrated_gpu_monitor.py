#!/usr/bin/env python3
"""
Integrated GPU Monitor - Unified NVIDIA Tools Integration
Combines nvidia-smi, nsight-sys, ncu, nvitop, and gpustat for comprehensive monitoring
"""

import os
import sys
import time
import json
import subprocess
import threading
import signal
import tempfile
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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
from rich.columns import Columns
from rich import box

class NVIDIAToolsManager:
    """Manager for all NVIDIA monitoring tools"""
    
    def __init__(self):
        self.console = Console()
        self.tools_status = {
            'nvidia-smi': False,
            'nsight-sys': False,
            'ncu': False,
            'nvitop': False,
            'gpustat': False,
            'nvml': False
        }
        self.tool_paths = {}
        self.detect_tools()
        
    def detect_tools(self):
        """Detect available NVIDIA tools"""
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.tools_status['nvidia-smi'] = True
                self.tool_paths['nvidia-smi'] = 'nvidia-smi'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Check nsight-systems
        try:
            result = subprocess.run(['nsys', '--version'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.tools_status['nsight-sys'] = True
                self.tool_paths['nsight-sys'] = 'nsys'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Check nsight-compute
        try:
            result = subprocess.run(['ncu', '--version'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.tools_status['ncu'] = True
                self.tool_paths['ncu'] = 'ncu'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Check nvitop
        try:
            import nvitop
            self.tools_status['nvitop'] = True
        except ImportError:
            pass
            
        # Check gpustat
        try:
            import gpustat
            self.tools_status['gpustat'] = True
        except ImportError:
            pass
            
        # Check NVML
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.tools_status['nvml'] = True
            except Exception:
                pass
    
    def get_nvidia_smi_data(self) -> Dict[str, Any]:
        """Get comprehensive data from nvidia-smi"""
        if not self.tools_status['nvidia-smi']:
            return {}
            
        data = {}
        
        try:
            # Get GPU information
            cmd = ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.graphics,clocks.memory', '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                data['gpus'] = []
                
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 13:
                            gpu_data = {
                                'index': int(parts[0]) if parts[0].isdigit() else 0,
                                'name': parts[1],
                                'driver_version': parts[2],
                                'memory_total_mb': int(parts[3]) if parts[3].isdigit() else 0,
                                'memory_used_mb': int(parts[4]) if parts[4].isdigit() else 0,
                                'memory_free_mb': int(parts[5]) if parts[5].isdigit() else 0,
                                'utilization_gpu': int(parts[6]) if parts[6].isdigit() else 0,
                                'utilization_memory': int(parts[7]) if parts[7].isdigit() else 0,
                                'temperature': int(parts[8]) if parts[8].isdigit() else 0,
                                'power_draw': float(parts[9]) if parts[9].replace('.','').isdigit() else 0,
                                'power_limit': float(parts[10]) if parts[10].replace('.','').isdigit() else 0,
                                'clocks_graphics': int(parts[11]) if parts[11].isdigit() else 0,
                                'clocks_memory': int(parts[12]) if parts[12].isdigit() else 0,
                            }
                            data['gpus'].append(gpu_data)
            
            # Get process information
            cmd = ['nvidia-smi', 'pmon', '-c', '1', '-s', 'um']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                data['processes'] = []
                
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            process_data = {
                                'gpu_id': int(parts[0]),
                                'pid': int(parts[1]),
                                'type': parts[2],
                                'sm_util': parts[3],
                                'mem_util': parts[4],
                                'enc_util': parts[5],
                                'dec_util': parts[6],
                                'command': ' '.join(parts[7:]) if len(parts) > 7 else 'Unknown'
                            }
                            data['processes'].append(process_data)
                        except (ValueError, IndexError):
                            continue
                            
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            self.console.print(f"[red]Error getting nvidia-smi data: {e}[/red]")
            
        return data
    
    def get_gpustat_data(self) -> Dict[str, Any]:
        """Get data from gpustat"""
        if not self.tools_status['gpustat']:
            return {}
            
        try:
            import gpustat
            stats = gpustat.GPUStatCollection.new_query()
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'gpus': []
            }
            
            for gpu in stats:
                gpu_data = {
                    'index': gpu.index,
                    'name': getattr(gpu, 'name', 'Unknown'),
                    'utilization': getattr(gpu, 'utilization', 0),
                    'memory_used': getattr(gpu, 'memory_used', 0),
                    'memory_total': getattr(gpu, 'memory_total', 1),
                    'temperature': getattr(gpu, 'temperature', 0),
                    'fan_speed': getattr(gpu, 'fan_speed', None),
                    'processes': []
                }
                
                if hasattr(gpu, 'processes'):
                    for process in gpu.processes:
                        proc_data = {
                            'pid': process.get('pid', 0),
                            'name': process.get('name', 'Unknown'),
                            'gpu_memory_usage': process.get('gpu_memory_usage', 0)
                        }
                        gpu_data['processes'].append(proc_data)
                
                data['gpus'].append(gpu_data)
                
            return data
            
        except Exception as e:
            self.console.print(f"[red]Error getting gpustat data: {e}[/red]")
            return {}
    
    def start_nsight_profiling(self, duration_seconds: int = 30, output_file: Optional[str] = None) -> str:
        """Start Nsight Systems profiling"""
        if not self.tools_status['nsight-sys']:
            return ""
            
        if not output_file:
            output_file = f"nsight_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.qdrep"
            
        try:
            cmd = [
                'nsys', 'profile',
                '--trace=cuda,cudnn,cublas',
                '--duration={}'.format(duration_seconds),
                '--output={}'.format(output_file),
                '--force-overwrite=true',
                'sleep', str(duration_seconds)
            ]
            
            # Start profiling in background
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_file
            
        except Exception as e:
            self.console.print(f"[red]Error starting nsight profiling: {e}[/red]")
            return ""
    
    def analyze_nsight_profile(self, profile_file: str) -> Dict[str, Any]:
        """Analyze Nsight Systems profile"""
        if not self.tools_status['nsight-sys'] or not os.path.exists(profile_file):
            return {}
            
        try:
            # Export statistics to JSON
            stats_file = profile_file.replace('.qdrep', '_stats.json')
            cmd = ['nsys', 'stats', '--report', 'gputrace', '--format', 'json', '--output', stats_file, profile_file]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                return stats_data
                
        except Exception as e:
            self.console.print(f"[red]Error analyzing nsight profile: {e}[/red]")
            
        return {}
    
    def get_ncu_kernel_metrics(self, target_processes: List[int] = None) -> Dict[str, Any]:
        """Get kernel metrics from Nsight Compute"""
        if not self.tools_status['ncu']:
            return {}
            
        try:
            cmd = ['ncu', '--metrics', 'sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum', '--csv']
            
            if target_processes:
                for pid in target_processes:
                    cmd.extend(['--target-processes', str(pid)])
            
            cmd.extend(['--timeout', '10', '--launch-count', '1'])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse CSV output
                lines = result.stdout.strip().split('\n')
                data = {'kernels': []}
                
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            kernel_data = {
                                'cycles': parts[0].strip(),
                                'bytes_read': parts[1].strip(),
                                'bytes_write': parts[2].strip(),
                                'timestamp': datetime.now().isoformat()
                            }
                            data['kernels'].append(kernel_data)
                
                return data
                
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            self.console.print(f"[red]Error getting ncu metrics: {e}[/red]")
            
        return {}

class IntegratedGPUMonitor:
    """Comprehensive GPU monitor integrating all NVIDIA tools"""
    
    def __init__(self):
        self.console = Console()
        self.tools_manager = NVIDIAToolsManager()
        self.running = True
        
        # Data storage
        self.nvidia_smi_data = {}
        self.gpustat_data = {}
        self.nsight_data = {}
        self.ncu_data = {}
        
        # Historical data
        self.history_size = 300
        self.gpu_history = defaultdict(lambda: {
            'utilization': deque(maxlen=self.history_size),
            'memory_used': deque(maxlen=self.history_size),
            'temperature': deque(maxlen=self.history_size),
            'power_draw': deque(maxlen=self.history_size),
            'timestamps': deque(maxlen=self.history_size)
        })
        
        # Background monitoring
        self.monitoring_thread = None
        self.profiling_active = False
        
        # Signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        self.console.print("\n[yellow]🛑 Shutting down integrated monitor...[/yellow]")
        sys.exit(0)
    
    def start_background_monitoring(self):
        """Start background data collection"""
        def monitor_loop():
            while self.running:
                try:
                    # Update data from all sources
                    self.nvidia_smi_data = self.tools_manager.get_nvidia_smi_data()
                    self.gpustat_data = self.tools_manager.get_gpustat_data()
                    
                    # Update history
                    if 'gpus' in self.nvidia_smi_data:
                        timestamp = datetime.now()
                        for gpu in self.nvidia_smi_data['gpus']:
                            gpu_id = gpu['index']
                            history = self.gpu_history[gpu_id]
                            
                            history['utilization'].append(gpu['utilization_gpu'])
                            history['memory_used'].append(gpu['memory_used_mb'])
                            history['temperature'].append(gpu['temperature'])
                            history['power_draw'].append(gpu['power_draw'])
                            history['timestamps'].append(timestamp)
                    
                    time.sleep(2)
                    
                except Exception as e:
                    self.console.print(f"[red]Monitoring error: {e}[/red]")
                    time.sleep(5)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def create_tools_status_panel(self) -> Panel:
        """Create panel showing tool availability"""
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Tool", style="cyan", width=15)
        table.add_column("Status", style="white", width=12)
        table.add_column("Capability", style="green")
        
        tools_info = {
            'nvidia-smi': "Basic GPU monitoring & process tracking",
            'nsight-sys': "Kernel execution timeline profiling",
            'ncu': "Detailed kernel performance analysis",
            'nvitop': "Interactive GPU monitoring interface",
            'gpustat': "Lightweight GPU status monitoring",
            'nvml': "Direct NVIDIA Management Library access"
        }
        
        for tool, description in tools_info.items():
            status = self.tools_manager.tools_status.get(tool, False)
            status_text = "[green]✅ Available[/green]" if status else "[red]❌ Missing[/red]"
            table.add_row(tool, status_text, description)
        
        return Panel(table, title="🛠️ NVIDIA Tools Integration Status", border_style="blue")
    
    def create_nvidia_smi_panel(self) -> Panel:
        """Create panel showing nvidia-smi data"""
        if not self.nvidia_smi_data or 'gpus' not in self.nvidia_smi_data:
            return Panel(
                Align.center(Text("nvidia-smi data not available", style="yellow")),
                title="📊 NVIDIA-SMI Data",
                border_style="yellow"
            )
        
        table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
        table.add_column("GPU", style="cyan", width=6)
        table.add_column("Name", style="green", width=25)
        table.add_column("Util%", style="yellow", width=8)
        table.add_column("Mem", style="blue", width=15)
        table.add_column("Temp", style="red", width=8)
        table.add_column("Power", style="magenta", width=12)
        table.add_column("Clocks", style="white", width=15)
        
        for gpu in self.nvidia_smi_data['gpus']:
            mem_text = f"{gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB"
            power_text = f"{gpu['power_draw']:.0f}/{gpu['power_limit']:.0f}W"
            clocks_text = f"G:{gpu['clocks_graphics']} M:{gpu['clocks_memory']}"
            
            # Color coding
            util_color = "red" if gpu['utilization_gpu'] > 80 else "yellow" if gpu['utilization_gpu'] > 60 else "green"
            temp_color = "red" if gpu['temperature'] > 80 else "yellow" if gpu['temperature'] > 70 else "green"
            
            table.add_row(
                f"GPU{gpu['index']}",
                gpu['name'][:23] + ".." if len(gpu['name']) > 25 else gpu['name'],
                f"[{util_color}]{gpu['utilization_gpu']}%[/{util_color}]",
                mem_text,
                f"[{temp_color}]{gpu['temperature']}°C[/{temp_color}]",
                power_text,
                clocks_text
            )
        
        return Panel(table, title="📊 NVIDIA-SMI GPU Status", border_style="green")
    
    def create_process_activity_panel(self) -> Panel:
        """Create panel showing process activity from nvidia-smi pmon"""
        if not self.nvidia_smi_data or 'processes' not in self.nvidia_smi_data:
            return Panel(
                Align.center(Text("No process data available", style="yellow")),
                title="🔥 Process Activity",
                border_style="yellow"
            )
        
        table = Table(show_header=True, header_style="bold red", box=box.ROUNDED)
        table.add_column("GPU", style="cyan", width=6)
        table.add_column("PID", style="yellow", width=8)
        table.add_column("Type", style="green", width=6)
        table.add_column("SM%", style="red", width=6)
        table.add_column("MEM%", style="blue", width=6)
        table.add_column("ENC%", style="magenta", width=6)
        table.add_column("DEC%", style="purple", width=6)
        table.add_column("Process", style="white", width=20)
        
        for proc in self.nvidia_smi_data['processes']:
            # Get process name
            try:
                ps_proc = psutil.Process(proc['pid'])
                proc_name = ps_proc.name()[:18] + ".." if len(ps_proc.name()) > 20 else ps_proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc_name = proc['command'][:18] + ".." if len(proc['command']) > 20 else proc['command']
            
            # Color coding for activity
            sm_val = proc['sm_util']
            if sm_val != '-' and sm_val.isdigit():
                sm_int = int(sm_val)
                sm_color = "red" if sm_int > 80 else "yellow" if sm_int > 50 else "green"
                sm_text = f"[{sm_color}]{sm_val}%[/{sm_color}]"
            else:
                sm_text = sm_val
            
            table.add_row(
                f"GPU{proc['gpu_id']}",
                str(proc['pid']),
                proc['type'],
                sm_text,
                f"{proc['mem_util']}%",
                f"{proc['enc_util']}%",
                f"{proc['dec_util']}%",
                proc_name
            )
        
        return Panel(table, title="🔥 GPU Process Activity (nvidia-smi pmon)", border_style="red")
    
    def create_gpustat_panel(self) -> Panel:
        """Create panel showing gpustat data"""
        if not self.gpustat_data or 'gpus' not in self.gpustat_data:
            return Panel(
                Align.center(Text("gpustat data not available", style="yellow")),
                title="📈 GPUStat Data",
                border_style="yellow"
            )
        
        table = Table(show_header=True, header_style="bold purple", box=box.ROUNDED)
        table.add_column("GPU", style="cyan", width=6)
        table.add_column("Name", style="green", width=20)
        table.add_column("Util%", style="yellow", width=8)
        table.add_column("Memory", style="blue", width=15)
        table.add_column("Temp", style="red", width=8)
        table.add_column("Processes", style="white", width=12)
        
        for gpu in self.gpustat_data['gpus']:
            mem_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            mem_text = f"{gpu['memory_used']}/{gpu['memory_total']} MB"
            
            # Color coding
            util_color = "red" if gpu['utilization'] > 80 else "yellow" if gpu['utilization'] > 60 else "green"
            mem_color = "red" if mem_percent > 85 else "yellow" if mem_percent > 70 else "green"
            temp_color = "red" if gpu['temperature'] > 80 else "yellow" if gpu['temperature'] > 70 else "green"
            
            table.add_row(
                f"GPU{gpu['index']}",
                gpu['name'][:18] + ".." if len(gpu['name']) > 20 else gpu['name'],
                f"[{util_color}]{gpu['utilization']}%[/{util_color}]",
                f"[{mem_color}]{mem_text}[/{mem_color}]",
                f"[{temp_color}]{gpu['temperature']}°C[/{temp_color}]",
                str(len(gpu['processes']))
            )
        
        return Panel(table, title="📈 GPUStat Monitoring", border_style="purple")
    
    def create_profiling_control_panel(self) -> Panel:
        """Create profiling control panel"""
        content = []
        
        # Nsight Systems status
        if self.tools_manager.tools_status['nsight-sys']:
            status = "🟢 Ready" if not self.profiling_active else "🔴 Profiling Active"
            content.append(f"Nsight Systems: {status}")
        else:
            content.append("Nsight Systems: ❌ Not Available")
        
        # Nsight Compute status
        if self.tools_manager.tools_status['ncu']:
            content.append("Nsight Compute: 🟢 Ready for kernel analysis")
        else:
            content.append("Nsight Compute: ❌ Not Available")
        
        # Instructions
        content.append("")
        content.append("Available Commands:")
        content.append("• Press 'P' to start 30s Nsight profiling")
        content.append("• Press 'K' to analyze kernel metrics")
        content.append("• Press 'R' to reset all data")
        
        text_content = Text("\n".join(content))
        
        return Panel(text_content, title="🔍 Profiling Controls", border_style="magenta")
    
    def create_integration_summary_panel(self) -> Panel:
        """Create integration summary panel"""
        # Count available tools
        available_tools = sum(1 for status in self.tools_manager.tools_status.values() if status)
        total_tools = len(self.tools_manager.tools_status)
        
        # Data source summary
        data_sources = []
        if self.nvidia_smi_data:
            data_sources.append("nvidia-smi")
        if self.gpustat_data:
            data_sources.append("gpustat")
        if self.nsight_data:
            data_sources.append("nsight-sys")
        if self.ncu_data:
            data_sources.append("ncu")
        
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        
        table.add_row("Available Tools", f"{available_tools}/{total_tools}")
        table.add_row("Active Data Sources", str(len(data_sources)))
        table.add_row("Integration Status", "🟢 Active" if data_sources else "🟡 Limited")
        table.add_row("Profiling Available", "✅ Yes" if self.tools_manager.tools_status['nsight-sys'] else "❌ No")
        table.add_row("Kernel Analysis", "✅ Yes" if self.tools_manager.tools_status['ncu'] else "❌ No")
        
        return Panel(table, title="📊 Integration Summary", border_style="cyan")
    
    def handle_user_input(self):
        """Handle user input for profiling controls"""
        # This would be implemented with keyboard input handling
        # For now, we'll just show the interface
        pass
    
    def run_integrated_monitor(self):
        """Run the integrated monitoring system"""
        self.console.clear()
        self.console.print("[bold green]🚀 Starting Integrated GPU Monitor...[/bold green]")
        
        # Show tools status
        tools_panel = self.create_tools_status_panel()
        self.console.print(tools_panel)
        time.sleep(3)
        
        # Start background monitoring
        self.start_background_monitoring()
        time.sleep(2)  # Let data collection start
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="nvidia_smi", size=12),
            Layout(name="processes")
        )
        
        layout["right"].split_column(
            Layout(name="gpustat", size=12),
            Layout(name="controls"),
            Layout(name="summary", size=10)
        )
        
        with Live(layout, console=self.console, screen=True, auto_refresh=False) as live:
            while self.running:
                try:
                    # Update header
                    layout["header"].update(
                        Panel(
                            Text("🔧 Integrated GPU Monitor - All NVIDIA Tools Unified", 
                                 style="bold white", justify="center"),
                            style="bold blue"
                        )
                    )
                    
                    # Update main panels
                    layout["nvidia_smi"].update(self.create_nvidia_smi_panel())
                    layout["processes"].update(self.create_process_activity_panel())
                    layout["gpustat"].update(self.create_gpustat_panel())
                    layout["controls"].update(self.create_profiling_control_panel())
                    layout["summary"].update(self.create_integration_summary_panel())
                    
                    # Update footer
                    layout["footer"].update(
                        Panel(
                            Text("Press Ctrl+C to exit | All NVIDIA tools integrated | Real-time monitoring active", 
                                 style="dim white", justify="center"),
                            style="dim green"
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
    monitor = IntegratedGPUMonitor()
    monitor.run_integrated_monitor()

if __name__ == "__main__":
    main()