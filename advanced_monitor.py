#!/usr/bin/env python3
"""
Advanced System Monitor with Alerts and Historical Data
Extended monitoring with logging, alerts, and performance tracking
"""

import time
import psutil
import GPUtil
import numpy as np
import pandas as pd
try:
    import pynvml
except ImportError:
    try:
        import nvidia_ml_py3 as pynvml
    except ImportError:
        pynvml = None
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.align import Align
from rich import box
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import threading
from collections import deque

class AdvancedSystemMonitor:
    def __init__(self, history_size=100, log_file="system_monitor.log"):
        self.console = Console()
        self.history_size = history_size
        self.log_file = log_file
        
        # Historical data storage
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
        self.network_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'gpu_percent': 90.0,
            'gpu_memory_percent': 95.0,
            'gpu_temperature': 85.0,
            'disk_percent': 95.0
        }
        
        # Alert tracking
        self.active_alerts = set()
        self.alert_history = []
        
        self.running = True
        
    def log_alert(self, alert_type, message, severity="WARNING"):
        """Log alerts to file and console"""
        timestamp = datetime.now()
        alert_data = {
            'timestamp': timestamp.isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message
        }
        
        self.alert_history.append(alert_data)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} [{severity}] {alert_type}: {message}\\n")
        
        # Show in console
        color = "red" if severity == "CRITICAL" else "yellow"
        self.console.print(f"[{color}]🚨 ALERT: {message}[/{color}]")
    
    def check_alerts(self, cpu_info, memory_info, gpu_info, disk_info):
        """Check for alert conditions"""
        current_alerts = set()
        
        # CPU alerts
        if cpu_info['avg_percent'] > self.alert_thresholds['cpu_percent']:
            alert_key = "cpu_high"
            current_alerts.add(alert_key)
            if alert_key not in self.active_alerts:
                self.log_alert("CPU_HIGH", f"CPU usage at {cpu_info['avg_percent']:.1f}%", "WARNING")
        
        # Memory alerts
        if memory_info['percent'] > self.alert_thresholds['memory_percent']:
            alert_key = "memory_high"
            current_alerts.add(alert_key)
            if alert_key not in self.active_alerts:
                self.log_alert("MEMORY_HIGH", f"Memory usage at {memory_info['percent']:.1f}%", "CRITICAL")
        
        # GPU alerts
        for i, gpu in enumerate(gpu_info):
            gpu_load_key = f"gpu_{i}_load"
            gpu_mem_key = f"gpu_{i}_memory"
            gpu_temp_key = f"gpu_{i}_temperature"
            
            if gpu['load'] > self.alert_thresholds['gpu_percent']:
                current_alerts.add(gpu_load_key)
                if gpu_load_key not in self.active_alerts:
                    self.log_alert("GPU_HIGH", f"GPU {i} ({gpu['name']}) load at {gpu['load']:.1f}%", "WARNING")
            
            if gpu['memory_percent'] > self.alert_thresholds['gpu_memory_percent']:
                current_alerts.add(gpu_mem_key)
                if gpu_mem_key not in self.active_alerts:
                    self.log_alert("GPU_MEMORY_HIGH", f"GPU {i} memory at {gpu['memory_percent']:.1f}%", "CRITICAL")
            
            if gpu['temperature'] > self.alert_thresholds['gpu_temperature']:
                current_alerts.add(gpu_temp_key)
                if gpu_temp_key not in self.active_alerts:
                    self.log_alert("GPU_TEMP_HIGH", f"GPU {i} temperature at {gpu['temperature']:.0f}°C", "CRITICAL")
        
        # Disk alerts
        for disk in disk_info:
            if disk['percent'] > self.alert_thresholds['disk_percent']:
                disk_key = f"disk_{disk['device']}"
                current_alerts.add(disk_key)
                if disk_key not in self.active_alerts:
                    self.log_alert("DISK_FULL", f"Disk {disk['device']} at {disk['percent']:.1f}%", "CRITICAL")
        
        self.active_alerts = current_alerts
    
    def get_enhanced_cpu_info(self):
        """Get enhanced CPU information with per-core details"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_times = psutil.cpu_times_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq(percpu=True) if hasattr(psutil.cpu_freq, 'percpu') else [psutil.cpu_freq()]
        
        return {
            'percent': cpu_percent,
            'avg_percent': sum(cpu_percent) / len(cpu_percent),
            'max_percent': max(cpu_percent),
            'min_percent': min(cpu_percent),
            'user': cpu_times.user,
            'system': cpu_times.system,
            'idle': cpu_times.idle,
            'iowait': getattr(cpu_times, 'iowait', 0),
            'frequencies': cpu_freq,
            'cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        }
    
    def get_process_info(self, top_n=10):
        """Get top processes by CPU and memory usage"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage
        cpu_top = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:top_n]
        
        # Sort by memory usage
        mem_top = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:top_n]
        
        return {'cpu_top': cpu_top, 'memory_top': mem_top}
    
    def create_process_panel(self, process_info):
        """Create top processes panel"""
        layout = Layout()
        layout.split_row(
            Layout(name="cpu_processes"),
            Layout(name="mem_processes")
        )
        
        # CPU processes table
        cpu_table = Table(title="🔥 Top CPU Processes", show_header=True, header_style="bold red")
        cpu_table.add_column("PID", style="cyan", width=8)
        cpu_table.add_column("Name", style="green", width=20)
        cpu_table.add_column("CPU%", justify="right", style="red")
        cpu_table.add_column("Status", style="yellow", width=10)
        
        for proc in process_info['cpu_top'][:5]:
            cpu_table.add_row(
                str(proc['pid']),
                proc['name'][:18] + ".." if len(proc['name']) > 20 else proc['name'],
                f"{proc['cpu_percent']:.1f}%",
                proc['status']
            )
        
        # Memory processes table
        mem_table = Table(title="🧠 Top Memory Processes", show_header=True, header_style="bold blue")
        mem_table.add_column("PID", style="cyan", width=8)
        mem_table.add_column("Name", style="green", width=20)
        mem_table.add_column("MEM%", justify="right", style="blue")
        mem_table.add_column("Status", style="yellow", width=10)
        
        for proc in process_info['memory_top'][:5]:
            mem_table.add_row(
                str(proc['pid']),
                proc['name'][:18] + ".." if len(proc['name']) > 20 else proc['name'],
                f"{proc['memory_percent']:.1f}%",
                proc['status']
            )
        
        layout["cpu_processes"].update(Panel(cpu_table, border_style="red"))
        layout["mem_processes"].update(Panel(mem_table, border_style="blue"))
        
        return Panel(layout, title="📊 Process Monitor", border_style="white")
    
    def create_alerts_panel(self):
        """Create alerts panel"""
        if not self.alert_history:
            return Panel(
                Align.center(Text("No alerts", style="green")),
                title="🚨 System Alerts",
                border_style="green"
            )
        
        table = Table(show_header=True, header_style="bold red")
        table.add_column("Time", style="cyan", width=12)
        table.add_column("Type", style="yellow", width=15)
        table.add_column("Message", style="white")
        table.add_column("Severity", justify="center", width=10)
        
        # Show last 5 alerts
        for alert in self.alert_history[-5:]:
            time_str = datetime.fromisoformat(alert['timestamp']).strftime("%H:%M:%S")
            severity_color = "red" if alert['severity'] == "CRITICAL" else "yellow"
            
            table.add_row(
                time_str,
                alert['type'],
                alert['message'],
                f"[{severity_color}]{alert['severity']}[/{severity_color}]"
            )
        
        border_color = "red" if any(a['severity'] == "CRITICAL" for a in self.alert_history[-5:]) else "yellow"
        return Panel(table, title="🚨 Recent Alerts", border_style=border_color)
    
    def create_performance_trends_panel(self):
        """Create performance trends panel"""
        if len(self.cpu_history) < 10:
            return Panel(
                Align.center(Text("Collecting data...", style="yellow")),
                title="📈 Performance Trends",
                border_style="yellow"
            )
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="green")
        table.add_column("Avg (5min)", style="blue")
        table.add_column("Max (5min)", style="red")
        table.add_column("Trend", justify="center")
        
        # CPU trends
        recent_cpu = list(self.cpu_history)[-30:]  # Last 30 readings (1 minute)
        cpu_avg = np.mean(recent_cpu) if recent_cpu else 0
        cpu_max = np.max(recent_cpu) if recent_cpu else 0
        cpu_current = recent_cpu[-1] if recent_cpu else 0
        
        # Simple trend calculation
        if len(recent_cpu) > 10:
            cpu_trend = "📈" if recent_cpu[-1] > np.mean(recent_cpu[-10:-5]) else "📉"
        else:
            cpu_trend = "➡️"
        
        table.add_row(
            "CPU Usage",
            f"{cpu_current:.1f}%",
            f"{cpu_avg:.1f}%",
            f"{cpu_max:.1f}%",
            cpu_trend
        )
        
        # Memory trends
        recent_mem = list(self.memory_history)[-30:]
        if recent_mem:
            mem_avg = np.mean(recent_mem)
            mem_max = np.max(recent_mem)
            mem_current = recent_mem[-1]
            mem_trend = "📈" if len(recent_mem) > 10 and recent_mem[-1] > np.mean(recent_mem[-10:-5]) else "📉"
            
            table.add_row(
                "Memory Usage",
                f"{mem_current:.1f}%",
                f"{mem_avg:.1f}%",
                f"{mem_max:.1f}%",
                mem_trend
            )
        
        return Panel(table, title="📈 Performance Trends", border_style="magenta")
    
    def update_history(self, cpu_info, memory_info, gpu_info):
        """Update historical data"""
        timestamp = datetime.now()
        
        self.timestamps.append(timestamp)
        self.cpu_history.append(cpu_info['avg_percent'])
        self.memory_history.append(memory_info['percent'])
        
        if gpu_info:
            gpu_avg_load = np.mean([gpu['load'] for gpu in gpu_info])
            self.gpu_history.append(gpu_avg_load)
        else:
            self.gpu_history.append(0)
    
    def save_performance_report(self):
        """Save performance report to file"""
        if not self.cpu_history:
            return
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_stats': {
                'average': float(np.mean(self.cpu_history)),
                'maximum': float(np.max(self.cpu_history)),
                'minimum': float(np.min(self.cpu_history))
            },
            'memory_stats': {
                'average': float(np.mean(self.memory_history)),
                'maximum': float(np.max(self.memory_history)),
                'minimum': float(np.min(self.memory_history))
            },
            'gpu_stats': {
                'average': float(np.mean(self.gpu_history)),
                'maximum': float(np.max(self.gpu_history)),
                'minimum': float(np.min(self.gpu_history))
            } if self.gpu_history and any(self.gpu_history) else None,
            'alerts_count': len(self.alert_history),
            'monitoring_duration_minutes': len(self.cpu_history) * 2 / 60  # 2-second intervals
        }
        
        with open('performance_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def get_gpu_info(self):
        """Get GPU information using GPUtil"""
        gpus = []
        try:
            gpu_list = GPUtil.getGPUs()
            for gpu in gpu_list:
                gpus.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                })
        except Exception:
            pass
        return gpus
    
    def get_memory_info(self):
        """Get memory information"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'used': memory.used,
            'percent': memory.percent,
            'available': memory.available
        }
    
    def get_disk_info(self):
        """Get disk information"""
        disks = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'total': usage.total,
                    'used': usage.used,
                    'percent': (usage.used / usage.total) * 100
                })
            except PermissionError:
                continue
        return disks
    
    def run_advanced_monitor(self):
        """Run the advanced monitoring system"""
        self.console.clear()
        self.console.print("[bold green]🚀 Starting Advanced System Monitor...[/bold green]")
        
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
            Layout(name="cpu", size=12),
            Layout(name="processes")
        )
        
        layout["right"].split_column(
            Layout(name="alerts", size=10),
            Layout(name="trends")
        )
        
        with Live(layout, console=self.console, screen=True, auto_refresh=False) as live:
            while self.running:
                try:
                    # Get system information
                    cpu_info = self.get_enhanced_cpu_info()
                    memory_info = self.get_memory_info()
                    gpu_info = self.get_gpu_info()
                    disk_info = self.get_disk_info()
                    process_info = self.get_process_info()
                    
                    # Update historical data
                    self.update_history(cpu_info, memory_info, gpu_info)
                    
                    # Check for alerts
                    self.check_alerts(cpu_info, memory_info, gpu_info, disk_info)
                    
                    # Update layout
                    layout["header"].update(
                        Panel(
                            Text("🔍 Advanced System Monitor - Enhanced Monitoring with Alerts & Trends", 
                                 style="bold white", justify="center"),
                            style="bold green"
                        )
                    )
                    
                    # CPU panel (simplified for space)
                    cpu_table = Table(show_header=True, header_style="bold cyan")
                    cpu_table.add_column("Metric", style="cyan")
                    cpu_table.add_column("Value", style="green")
                    
                    cpu_table.add_row("Average CPU", f"{cpu_info['avg_percent']:.1f}%")
                    cpu_table.add_row("Max Core", f"{cpu_info['max_percent']:.1f}%")
                    cpu_table.add_row("User/System", f"{cpu_info['user']:.1f}% / {cpu_info['system']:.1f}%")
                    cpu_table.add_row("Memory", f"{memory_info['percent']:.1f}%")
                    
                    if gpu_info:
                        gpu_avg = np.mean([gpu['load'] for gpu in gpu_info])
                        cpu_table.add_row("GPU Average", f"{gpu_avg:.1f}%")
                    
                    layout["cpu"].update(Panel(cpu_table, title="💻 System Overview", border_style="cyan"))
                    layout["processes"].update(self.create_process_panel(process_info))
                    layout["alerts"].update(self.create_alerts_panel())
                    layout["trends"].update(self.create_performance_trends_panel())
                    
                    layout["footer"].update(
                        Panel(
                            Text("Press Ctrl+C to exit and save report | 📊 Data points collected: " + 
                                 str(len(self.cpu_history)), style="dim white", justify="center"),
                            style="dim blue"
                        )
                    )
                    
                    live.refresh()
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(2)
        
        # Save final report
        self.save_performance_report()
        self.console.print("[green]Performance report saved to performance_report.json[/green]")

def main():
    monitor = AdvancedSystemMonitor()
    monitor.run_advanced_monitor()

if __name__ == "__main__":
    main()