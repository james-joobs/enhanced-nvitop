#!/usr/bin/env python3
"""
Advanced GPU Monitor - Dedicated NVIDIA GPU Monitoring
Comprehensive GPU analytics, memory tracking, and performance monitoring
"""

import time
import psutil
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
import signal
import sys
import subprocess
from typing import Dict, List, Optional, Any

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

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.align import Align
from rich.columns import Columns
from rich import box
from rich.tree import Tree

class DetailedGPUInfo:
    """Container for detailed GPU information"""
    def __init__(self):
        self.device_id: int = 0
        self.name: str = ""
        self.driver_version: str = ""
        self.pci_bus: str = ""
        
        # Performance metrics
        self.gpu_utilization: float = 0.0
        self.memory_utilization: float = 0.0
        self.encoder_utilization: float = 0.0
        self.decoder_utilization: float = 0.0
        
        # Memory information
        self.memory_total: int = 0
        self.memory_used: int = 0
        self.memory_free: int = 0
        self.memory_percent: float = 0.0
        
        # Temperature and power
        self.temperature: int = 0
        self.temperature_max: int = 0
        self.temperature_slowdown: int = 0
        self.power_draw: float = 0.0
        self.power_limit: float = 0.0
        self.power_percent: float = 0.0
        
        # Clock speeds
        self.clock_graphics: int = 0
        self.clock_memory: int = 0
        self.clock_sm: int = 0
        self.clock_graphics_max: int = 0
        self.clock_memory_max: int = 0
        
        # Fan and throttling
        self.fan_speed: int = 0
        self.performance_state: str = ""
        self.throttle_reasons: List[str] = []
        
        # Processes
        self.processes: List[Dict[str, Any]] = []
        
        # Compute capability
        self.compute_capability: tuple = (0, 0)
        self.cuda_cores: int = 0

class AdvancedGPUMonitor:
    def __init__(self, history_size=300):  # 10 minutes at 2-second intervals
        self.console = Console()
        self.history_size = history_size
        self.running = True
        
        # GPU data storage
        self.gpu_data: Dict[int, DetailedGPUInfo] = {}
        self.gpu_history: Dict[int, Dict[str, deque]] = defaultdict(lambda: {
            'utilization': deque(maxlen=history_size),
            'memory_percent': deque(maxlen=history_size),
            'temperature': deque(maxlen=history_size),
            'power_draw': deque(maxlen=history_size),
            'clock_graphics': deque(maxlen=history_size),
            'clock_memory': deque(maxlen=history_size),
            'timestamps': deque(maxlen=history_size)
        })
        
        # Alert thresholds specifically for GPU
        self.gpu_alert_thresholds = {
            'temperature_warning': 80,
            'temperature_critical': 85,
            'memory_warning': 85,
            'memory_critical': 95,
            'power_warning': 90,
            'utilization_sustained_high': 95,  # Sustained high usage
            'memory_fragmentation_warning': 80
        }
        
        # Performance tracking
        self.performance_stats = defaultdict(lambda: {
            'peak_utilization': 0,
            'peak_memory': 0,
            'peak_temperature': 0,
            'peak_power': 0,
            'total_energy': 0,  # kWh
            'operating_time': 0,  # seconds
            'thermal_throttle_events': 0,
            'power_throttle_events': 0
        })
        
        # Initialize NVIDIA ML
        self.nvml_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.console.print(f"[green]✅ Initialized NVIDIA ML - Found {self.gpu_count} GPU(s)[/green]")
            except Exception as e:
                self.console.print(f"[red]❌ Failed to initialize NVIDIA ML: {e}[/red]")
        else:
            self.console.print("[red]❌ NVIDIA ML library not available[/red]")
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.running = False
        self.save_performance_report()
        self.console.print("\n[yellow]🛑 Shutting down GPU monitor...[/yellow]")
        sys.exit(0)
    
    def get_gpu_processes(self, device_id: int) -> List[Dict[str, Any]]:
        """Get detailed process information for a GPU"""
        if not self.nvml_available:
            return []
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            
            detailed_processes = []
            for proc in processes:
                try:
                    # Get process info
                    proc_info = {
                        'pid': proc.pid,
                        'memory_used': proc.usedGpuMemory,
                        'name': 'Unknown',
                        'command': 'Unknown'
                    }
                    
                    # Try to get process name and command
                    try:
                        ps_proc = psutil.Process(proc.pid)
                        proc_info['name'] = ps_proc.name()
                        proc_info['command'] = ' '.join(ps_proc.cmdline()[:3])  # First 3 args
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    detailed_processes.append(proc_info)
                except Exception:
                    continue
            
            return detailed_processes
        except Exception:
            return []
    
    def get_detailed_gpu_info(self, device_id: int) -> DetailedGPUInfo:
        """Get comprehensive GPU information"""
        gpu_info = DetailedGPUInfo()
        gpu_info.device_id = device_id
        
        if not self.nvml_available:
            return gpu_info
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # Basic device info
            name_result = pynvml.nvmlDeviceGetName(handle)
            gpu_info.name = name_result.decode('utf-8') if isinstance(name_result, bytes) else str(name_result)
            
            driver_result = pynvml.nvmlSystemGetDriverVersion()
            gpu_info.driver_version = driver_result.decode('utf-8') if isinstance(driver_result, bytes) else str(driver_result)
            
            try:
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                bus_id = pci_info.busId
                gpu_info.pci_bus = bus_id.decode('utf-8') if isinstance(bus_id, bytes) else str(bus_id)
            except Exception:
                gpu_info.pci_bus = "Unknown"
            
            # Utilization
            try:
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info.gpu_utilization = util_info.gpu
                gpu_info.memory_utilization = util_info.memory
            except Exception:
                pass
            
            # Encoder/Decoder utilization
            try:
                enc_util, _ = pynvml.nvmlDeviceGetEncoderUtilization(handle)
                gpu_info.encoder_utilization = enc_util
            except Exception:
                pass
            
            try:
                dec_util, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
                gpu_info.decoder_utilization = dec_util
            except Exception:
                pass
            
            # Memory info
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info.memory_total = mem_info.total
                gpu_info.memory_used = mem_info.used
                gpu_info.memory_free = mem_info.free
                gpu_info.memory_percent = (mem_info.used / mem_info.total) * 100
            except Exception:
                pass
            
            # Temperature
            try:
                gpu_info.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_info.temperature_max = pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
                gpu_info.temperature_slowdown = pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
            except Exception:
                pass
            
            # Power
            try:
                gpu_info.power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                gpu_info.power_limit = pynvml.nvmlDeviceGetPowerManagementLimitDefault(handle) / 1000.0
                gpu_info.power_percent = (gpu_info.power_draw / gpu_info.power_limit) * 100 if gpu_info.power_limit > 0 else 0
            except Exception:
                pass
            
            # Clock speeds
            try:
                gpu_info.clock_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                gpu_info.clock_memory = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                gpu_info.clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                
                gpu_info.clock_graphics_max = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                gpu_info.clock_memory_max = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                pass
            
            # Fan speed
            try:
                gpu_info.fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                pass
            
            # Performance state
            try:
                pstate = pynvml.nvmlDeviceGetPerformanceState(handle)
                gpu_info.performance_state = f"P{pstate}"
            except Exception:
                pass
            
            # Throttle reasons
            try:
                throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                gpu_info.throttle_reasons = []
                if throttle_reasons & pynvml.nvmlClocksThrottleReasonGpuIdle:
                    gpu_info.throttle_reasons.append("GPU Idle")
                if throttle_reasons & pynvml.nvmlClocksThrottleReasonApplicationsClocksSetting:
                    gpu_info.throttle_reasons.append("App Clock Limit")
                if throttle_reasons & pynvml.nvmlClocksThrottleReasonSwPowerCap:
                    gpu_info.throttle_reasons.append("SW Power Limit")
                if throttle_reasons & pynvml.nvmlClocksThrottleReasonHwSlowdown:
                    gpu_info.throttle_reasons.append("HW Slowdown")
                if throttle_reasons & pynvml.nvmlClocksThrottleReasonSyncBoost:
                    gpu_info.throttle_reasons.append("Sync Boost")
                if throttle_reasons & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown:
                    gpu_info.throttle_reasons.append("Thermal")
                if throttle_reasons & pynvml.nvmlClocksThrottleReasonHwThermalSlowdown:
                    gpu_info.throttle_reasons.append("HW Thermal")
                if throttle_reasons & pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown:
                    gpu_info.throttle_reasons.append("Power Brake")
            except Exception:
                pass
            
            # Compute capability
            try:
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                gpu_info.compute_capability = (major, minor)
            except Exception:
                pass
            
            # Get processes
            gpu_info.processes = self.get_gpu_processes(device_id)
            
        except Exception as e:
            self.console.print(f"[red]Error getting GPU {device_id} info: {e}[/red]")
        
        return gpu_info
    
    def update_gpu_data(self):
        """Update GPU data for all devices"""
        if not self.nvml_available:
            return
        
        timestamp = datetime.now()
        
        for gpu_id in range(self.gpu_count):
            gpu_info = self.get_detailed_gpu_info(gpu_id)
            self.gpu_data[gpu_id] = gpu_info
            
            # Update history
            history = self.gpu_history[gpu_id]
            history['utilization'].append(gpu_info.gpu_utilization)
            history['memory_percent'].append(gpu_info.memory_percent)
            history['temperature'].append(gpu_info.temperature)
            history['power_draw'].append(gpu_info.power_draw)
            history['clock_graphics'].append(gpu_info.clock_graphics)
            history['clock_memory'].append(gpu_info.clock_memory)
            history['timestamps'].append(timestamp)
            
            # Update performance stats
            stats = self.performance_stats[gpu_id]
            stats['peak_utilization'] = max(stats['peak_utilization'], gpu_info.gpu_utilization)
            stats['peak_memory'] = max(stats['peak_memory'], gpu_info.memory_percent)
            stats['peak_temperature'] = max(stats['peak_temperature'], gpu_info.temperature)
            stats['peak_power'] = max(stats['peak_power'], gpu_info.power_draw)
            stats['operating_time'] += 2  # 2-second intervals
            
            # Energy calculation (approximate)
            stats['total_energy'] += (gpu_info.power_draw * 2) / 3600000  # Convert to kWh
            
            # Throttle event detection
            if 'Thermal' in gpu_info.throttle_reasons:
                stats['thermal_throttle_events'] += 1
            if any(reason in gpu_info.throttle_reasons for reason in ['SW Power Limit', 'Power Brake']):
                stats['power_throttle_events'] += 1
    
    def create_gpu_overview_panel(self) -> Panel:
        """Create GPU overview panel"""
        if not self.gpu_data:
            return Panel(
                Align.center(Text("No GPU data available", style="red")),
                title="🎮 GPU Overview",
                border_style="red"
            )
        
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("GPU", style="cyan", width=8)
        table.add_column("Name", style="green", width=25)
        table.add_column("Util%", justify="center", width=8)
        table.add_column("Mem%", justify="center", width=8)
        table.add_column("Temp", justify="center", width=8)
        table.add_column("Power", justify="center", width=10)
        table.add_column("PState", justify="center", width=8)
        table.add_column("Status", justify="center", width=12)
        
        for gpu_id, gpu_info in self.gpu_data.items():
            # Status determination
            status_items = []
            status_color = "green"
            
            if gpu_info.temperature >= self.gpu_alert_thresholds['temperature_critical']:
                status_items.append("🔥 HOT")
                status_color = "red"
            elif gpu_info.temperature >= self.gpu_alert_thresholds['temperature_warning']:
                status_items.append("⚠️ WARM")
                status_color = "yellow"
            
            if gpu_info.memory_percent >= self.gpu_alert_thresholds['memory_critical']:
                status_items.append("💾 FULL")
                status_color = "red"
            elif gpu_info.memory_percent >= self.gpu_alert_thresholds['memory_warning']:
                status_items.append("💾 HIGH")
                status_color = "yellow"
            
            if gpu_info.throttle_reasons:
                status_items.append("⏸️ THR")
                if status_color == "green":
                    status_color = "yellow"
            
            if not status_items:
                status_items.append("✅ OK")
            
            status_text = " ".join(status_items[:2])  # Max 2 status items
            
            # Color coding for utilization
            util_color = "red" if gpu_info.gpu_utilization > 90 else "yellow" if gpu_info.gpu_utilization > 70 else "green"
            mem_color = "red" if gpu_info.memory_percent > 90 else "yellow" if gpu_info.memory_percent > 70 else "green"
            temp_color = "red" if gpu_info.temperature > 80 else "yellow" if gpu_info.temperature > 70 else "green"
            
            table.add_row(
                f"GPU{gpu_id}",
                gpu_info.name[:23] + ".." if len(gpu_info.name) > 25 else gpu_info.name,
                f"[{util_color}]{gpu_info.gpu_utilization:.0f}%[/{util_color}]",
                f"[{mem_color}]{gpu_info.memory_percent:.0f}%[/{mem_color}]",
                f"[{temp_color}]{gpu_info.temperature}°C[/{temp_color}]",
                f"{gpu_info.power_draw:.0f}W",
                gpu_info.performance_state,
                f"[{status_color}]{status_text}[/{status_color}]"
            )
        
        return Panel(table, title="🎮 GPU Overview", border_style="blue")
    
    def create_gpu_details_panel(self, gpu_id: int) -> Panel:
        """Create detailed panel for specific GPU"""
        if gpu_id not in self.gpu_data:
            return Panel(
                Align.center(Text(f"GPU {gpu_id} not found", style="red")),
                title=f"🎯 GPU {gpu_id} Details",
                border_style="red"
            )
        
        gpu_info = self.gpu_data[gpu_id]
        
        # Create a layout with multiple sections
        layout = Layout()
        layout.split_column(
            Layout(name="specs", size=8),
            Layout(name="performance"),
            Layout(name="processes", size=8)
        )
        
        # Specifications table
        spec_table = Table(show_header=False, box=box.SIMPLE)
        spec_table.add_column("Property", style="cyan", width=18)
        spec_table.add_column("Value", style="white")
        
        spec_table.add_row("Name", gpu_info.name)
        spec_table.add_row("Driver Version", gpu_info.driver_version)
        spec_table.add_row("PCI Bus", gpu_info.pci_bus)
        spec_table.add_row("Compute Cap.", f"{gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}")
        spec_table.add_row("Memory Total", f"{gpu_info.memory_total / 1024**3:.1f} GB")
        
        layout["specs"].update(Panel(spec_table, title="Specifications", border_style="cyan"))
        
        # Performance metrics
        perf_layout = Layout()
        perf_layout.split_row(
            Layout(name="utilization"),
            Layout(name="clocks_power")
        )
        
        # Utilization table
        util_table = Table(show_header=False, box=box.SIMPLE)
        util_table.add_column("Metric", style="yellow", width=12)
        util_table.add_column("Value", style="white", width=8)
        util_table.add_column("Bar", style="white")
        
        # Create progress bars
        gpu_bar = "█" * int(gpu_info.gpu_utilization / 10) + "░" * (10 - int(gpu_info.gpu_utilization / 10))
        mem_bar = "█" * int(gpu_info.memory_percent / 10) + "░" * (10 - int(gpu_info.memory_percent / 10))
        temp_bar = "█" * int(gpu_info.temperature / 10) + "░" * (10 - int(gpu_info.temperature / 10))
        power_bar = "█" * int(gpu_info.power_percent / 10) + "░" * (10 - int(gpu_info.power_percent / 10))
        
        util_table.add_row("GPU", f"{gpu_info.gpu_utilization:.0f}%", f"[green]{gpu_bar}[/green]")
        util_table.add_row("Memory", f"{gpu_info.memory_percent:.0f}%", f"[blue]{mem_bar}[/blue]")
        util_table.add_row("Temp", f"{gpu_info.temperature}°C", f"[red]{temp_bar}[/red]")
        util_table.add_row("Power", f"{gpu_info.power_percent:.0f}%", f"[yellow]{power_bar}[/yellow]")
        
        perf_layout["utilization"].update(Panel(util_table, title="Utilization", border_style="green"))
        
        # Clocks and power table
        clock_table = Table(show_header=False, box=box.SIMPLE)
        clock_table.add_column("Metric", style="magenta", width=15)
        clock_table.add_column("Current", style="white", width=8)
        clock_table.add_column("Max", style="dim white", width=8)
        
        clock_table.add_row("Graphics Clock", f"{gpu_info.clock_graphics} MHz", f"{gpu_info.clock_graphics_max} MHz")
        clock_table.add_row("Memory Clock", f"{gpu_info.clock_memory} MHz", f"{gpu_info.clock_memory_max} MHz")
        clock_table.add_row("SM Clock", f"{gpu_info.clock_sm} MHz", "")
        clock_table.add_row("Power Draw", f"{gpu_info.power_draw:.1f} W", f"{gpu_info.power_limit:.1f} W")
        clock_table.add_row("Fan Speed", f"{gpu_info.fan_speed}%", "")
        
        perf_layout["clocks_power"].update(Panel(clock_table, title="Clocks & Power", border_style="magenta"))
        
        layout["performance"].update(perf_layout)
        
        # Processes table
        if gpu_info.processes:
            proc_table = Table(show_header=True, header_style="bold red", box=box.SIMPLE)
            proc_table.add_column("PID", style="cyan", width=8)
            proc_table.add_column("Process", style="green", width=20)
            proc_table.add_column("GPU Mem", style="yellow", width=10)
            
            for proc in gpu_info.processes[:5]:  # Show top 5 processes
                proc_table.add_row(
                    str(proc['pid']),
                    proc['name'][:18] + ".." if len(proc['name']) > 20 else proc['name'],
                    f"{proc['memory_used'] / 1024**2:.0f} MB"
                )
        else:
            proc_table = Align.center(Text("No GPU processes", style="dim"))
        
        layout["processes"].update(Panel(proc_table, title="GPU Processes", border_style="red"))
        
        return Panel(layout, title=f"🎯 GPU {gpu_id} - {gpu_info.name}", border_style="blue")
    
    def create_performance_history_panel(self, gpu_id: int) -> Panel:
        """Create performance history panel with trends"""
        if gpu_id not in self.gpu_history or len(self.gpu_history[gpu_id]['timestamps']) < 5:
            return Panel(
                Align.center(Text("Collecting data...", style="yellow")),
                title=f"📈 GPU {gpu_id} Trends",
                border_style="yellow"
            )
        
        history = self.gpu_history[gpu_id]
        
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Current", style="green", width=8)
        table.add_column("1min Avg", style="blue", width=8)
        table.add_column("5min Avg", style="blue", width=8)
        table.add_column("Peak", style="red", width=8)
        table.add_column("Trend", justify="center", width=6)
        
        # Calculate averages
        recent_1min = list(history['utilization'])[-30:]  # Last 1 minute
        recent_5min = list(history['utilization'])[-150:]  # Last 5 minutes
        
        if recent_1min:
            util_1min = np.mean(recent_1min)
            util_5min = np.mean(recent_5min) if len(recent_5min) >= 30 else np.mean(recent_1min)
            util_current = recent_1min[-1]
            util_peak = np.max(recent_5min) if recent_5min else util_current
            util_trend = "📈" if len(recent_1min) > 5 and util_current > np.mean(recent_1min[-10:-5]) else "📉"
            
            table.add_row(
                "GPU Util",
                f"{util_current:.0f}%",
                f"{util_1min:.0f}%",
                f"{util_5min:.0f}%",
                f"{util_peak:.0f}%",
                util_trend
            )
        
        # Memory trends
        recent_mem_1min = list(history['memory_percent'])[-30:]
        if recent_mem_1min:
            mem_1min = np.mean(recent_mem_1min)
            mem_current = recent_mem_1min[-1]
            mem_trend = "📈" if len(recent_mem_1min) > 5 and mem_current > np.mean(recent_mem_1min[-10:-5]) else "📉"
            
            table.add_row(
                "Memory",
                f"{mem_current:.0f}%",
                f"{mem_1min:.0f}%",
                "-",
                f"{np.max(recent_mem_1min):.0f}%",
                mem_trend
            )
        
        # Temperature trends
        recent_temp_1min = list(history['temperature'])[-30:]
        if recent_temp_1min:
            temp_1min = np.mean(recent_temp_1min)
            temp_current = recent_temp_1min[-1]
            temp_trend = "📈" if len(recent_temp_1min) > 5 and temp_current > np.mean(recent_temp_1min[-10:-5]) else "📉"
            
            table.add_row(
                "Temperature",
                f"{temp_current:.0f}°C",
                f"{temp_1min:.0f}°C",
                "-",
                f"{np.max(recent_temp_1min):.0f}°C",
                temp_trend
            )
        
        # Power trends
        recent_power_1min = list(history['power_draw'])[-30:]
        if recent_power_1min:
            power_1min = np.mean(recent_power_1min)
            power_current = recent_power_1min[-1]
            power_trend = "📈" if len(recent_power_1min) > 5 and power_current > np.mean(recent_power_1min[-10:-5]) else "📉"
            
            table.add_row(
                "Power",
                f"{power_current:.0f}W",
                f"{power_1min:.0f}W",
                "-",
                f"{np.max(recent_power_1min):.0f}W",
                power_trend
            )
        
        return Panel(table, title=f"📈 GPU {gpu_id} Performance Trends", border_style="magenta")
    
    def create_alerts_panel(self) -> Panel:
        """Create alerts panel for GPU-specific issues"""
        alerts = []
        
        for gpu_id, gpu_info in self.gpu_data.items():
            # Temperature alerts
            if gpu_info.temperature >= self.gpu_alert_thresholds['temperature_critical']:
                alerts.append({
                    'severity': 'CRITICAL',
                    'gpu': gpu_id,
                    'message': f"Temperature critical: {gpu_info.temperature}°C",
                    'color': 'red'
                })
            elif gpu_info.temperature >= self.gpu_alert_thresholds['temperature_warning']:
                alerts.append({
                    'severity': 'WARNING',
                    'gpu': gpu_id,
                    'message': f"Temperature high: {gpu_info.temperature}°C",
                    'color': 'yellow'
                })
            
            # Memory alerts
            if gpu_info.memory_percent >= self.gpu_alert_thresholds['memory_critical']:
                alerts.append({
                    'severity': 'CRITICAL',
                    'gpu': gpu_id,
                    'message': f"Memory critical: {gpu_info.memory_percent:.0f}%",
                    'color': 'red'
                })
            
            # Throttling alerts
            if gpu_info.throttle_reasons:
                alerts.append({
                    'severity': 'WARNING',
                    'gpu': gpu_id,
                    'message': f"Throttling: {', '.join(gpu_info.throttle_reasons[:2])}",
                    'color': 'yellow'
                })
        
        if not alerts:
            return Panel(
                Align.center(Text("All systems normal", style="green bold")),
                title="🚨 GPU Alerts",
                border_style="green"
            )
        
        table = Table(show_header=True, header_style="bold red", box=box.ROUNDED)
        table.add_column("GPU", style="cyan", width=6)
        table.add_column("Severity", style="white", width=10)
        table.add_column("Alert", style="white")
        
        for alert in alerts[-10:]:  # Show last 10 alerts
            table.add_row(
                f"GPU{alert['gpu']}",
                f"[{alert['color']}]{alert['severity']}[/{alert['color']}]",
                alert['message']
            )
        
        border_color = "red" if any(a['severity'] == 'CRITICAL' for a in alerts) else "yellow"
        return Panel(table, title="🚨 GPU Alerts", border_style=border_color)
    
    def save_performance_report(self):
        """Save detailed performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_duration_seconds': len(list(self.gpu_history.values())[0]['timestamps']) * 2 if self.gpu_history else 0,
            'gpu_count': len(self.gpu_data),
            'gpus': {}
        }
        
        for gpu_id in self.gpu_data:
            stats = self.performance_stats[gpu_id]
            gpu_data = self.gpu_data[gpu_id]
            
            report['gpus'][gpu_id] = {
                'name': gpu_data.name,
                'driver_version': gpu_data.driver_version,
                'performance_stats': dict(stats),
                'current_state': {
                    'utilization': gpu_data.gpu_utilization,
                    'memory_percent': gpu_data.memory_percent,
                    'temperature': gpu_data.temperature,
                    'power_draw': gpu_data.power_draw,
                    'throttle_reasons': gpu_data.throttle_reasons
                }
            }
        
        with open('gpu_performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    def run_gpu_monitor(self):
        """Run the advanced GPU monitoring system"""
        if not self.nvml_available:
            self.console.print("[red]❌ NVIDIA ML not available. Cannot monitor GPUs.[/red]")
            return
        
        self.console.clear()
        self.console.print("[bold green]🚀 Starting Advanced GPU Monitor...[/bold green]")
        time.sleep(1)
        
        # Determine layout based on GPU count
        if self.gpu_count == 1:
            # Single GPU - detailed view
            self.run_single_gpu_monitor()
        else:
            # Multiple GPUs - overview + rotating detailed view
            self.run_multi_gpu_monitor()
    
    def run_single_gpu_monitor(self):
        """Monitor single GPU with detailed view"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        layout["left"].split_column(
            Layout(name="overview", size=8),
            Layout(name="details")
        )
        
        layout["right"].split_column(
            Layout(name="trends"),
            Layout(name="alerts", size=12)
        )
        
        with Live(layout, console=self.console, screen=True, auto_refresh=False) as live:
            while self.running:
                try:
                    self.update_gpu_data()
                    
                    # Update layout
                    layout["header"].update(
                        Panel(
                            Text("🎮 Advanced GPU Monitor - Detailed NVIDIA GPU Analytics", 
                                 style="bold white", justify="center"),
                            style="bold blue"
                        )
                    )
                    
                    layout["overview"].update(self.create_gpu_overview_panel())
                    layout["details"].update(self.create_gpu_details_panel(0))
                    layout["trends"].update(self.create_performance_history_panel(0))
                    layout["alerts"].update(self.create_alerts_panel())
                    
                    layout["footer"].update(
                        Panel(
                            Text(f"Press Ctrl+C to exit and save report | Monitoring: {len(list(self.gpu_history[0]['timestamps']))} data points | Update: 2s", 
                                 style="dim white", justify="center"),
                            style="dim blue"
                        )
                    )
                    
                    live.refresh()
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(2)
    
    def run_multi_gpu_monitor(self):
        """Monitor multiple GPUs with rotating detailed view"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="overview", size=12),
            Layout(name="details"),
            Layout(name="footer", size=4)
        )
        
        layout["details"].split_row(
            Layout(name="gpu_detail", ratio=2),
            Layout(name="alerts_trends", ratio=1)
        )
        
        layout["alerts_trends"].split_column(
            Layout(name="alerts"),
            Layout(name="trends")
        )
        
        current_gpu = 0
        update_count = 0
        
        with Live(layout, console=self.console, screen=True, auto_refresh=False) as live:
            while self.running:
                try:
                    self.update_gpu_data()
                    
                    # Update layout
                    layout["header"].update(
                        Panel(
                            Text(f"🎮 Advanced GPU Monitor - {self.gpu_count} GPUs", 
                                 style="bold white", justify="center"),
                            style="bold blue"
                        )
                    )
                    
                    layout["overview"].update(self.create_gpu_overview_panel())
                    layout["gpu_detail"].update(self.create_gpu_details_panel(current_gpu))
                    layout["alerts"].update(self.create_alerts_panel())
                    layout["trends"].update(self.create_performance_history_panel(current_gpu))
                    
                    # Rotate through GPUs every 30 seconds (15 updates)
                    if update_count % 15 == 0:
                        current_gpu = (current_gpu + 1) % self.gpu_count
                    
                    layout["footer"].update(
                        Panel(
                            Text(f"Press Ctrl+C to exit | Detailed view: GPU {current_gpu} | Next rotation in {15 - (update_count % 15)} updates", 
                                 style="dim white", justify="center"),
                            style="dim blue"
                        )
                    )
                    
                    live.refresh()
                    time.sleep(2)
                    update_count += 1
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(2)

def main():
    """Main entry point"""
    monitor = AdvancedGPUMonitor()
    monitor.run_gpu_monitor()

if __name__ == "__main__":
    main()