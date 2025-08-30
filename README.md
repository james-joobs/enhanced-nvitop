# 🚀 Enhanced GPU & CPU Monitor

A powerful, colorful, and insightful monitoring system for NVIDIA GPUs and CPU systems, built on top of nvitop with enhanced features.

## Features

### 🌈 Enhanced Monitor (`enhanced_monitor.py`)
- **Real-time dashboard** with colorful Rich console output
- **CPU monitoring** with per-core usage, temperature, and frequency
- **GPU monitoring** with detailed memory, temperature, and power usage
- **Memory tracking** with RAM and swap usage
- **System information** including uptime and process count
- **Color-coded alerts** with status indicators

### 🔍 Advanced Monitor (`advanced_monitor.py`)
- **All enhanced monitor features** plus:
- **Alert system** with configurable thresholds
- **Performance trends** and historical data tracking
- **Top processes** monitoring (CPU and memory usage)
- **Logging system** with alert history
- **Performance reports** saved to JSON files
- **Real-time trend analysis** with visual indicators

### 🎯 Original nvitop
- Basic NVIDIA GPU monitoring using the original nvitop tool
- Simple and lightweight

## Installation

The project is configured with modern Python (3.11+) and includes all necessary dependencies:

```bash
# Dependencies are automatically managed with uv
uv sync
```

## Usage

### Interactive Menu
Run the main launcher for an interactive menu:
```bash
uv run python monitor.py
```

### Direct Mode Selection
Run specific monitors directly:

```bash
# Enhanced colorful monitor
uv run python monitor.py enhanced

# Advanced monitor with alerts
uv run python monitor.py advanced

# Dedicated GPU monitoring
uv run python monitor.py gpu

# Quick system overview
uv run python monitor.py overview

# Original nvitop
uv run python monitor.py nvitop
```

### Individual Scripts
You can also run the monitors directly:

```bash
# Enhanced monitor
uv run python enhanced_monitor.py

# Advanced monitor with alerts
uv run python advanced_monitor.py

# GPU Monitor Pro
uv run python gpu_monitor_advanced.py

# Original nvitop
uv run nvitop
```

## Monitor Types

### 1. Enhanced Monitor
Perfect for real-time monitoring with beautiful visuals:
- **CPU Panel**: Per-core usage, frequency, temperature, load average
- **Memory Panel**: RAM and swap usage with color-coded warnings
- **GPU Panel**: Multiple GPU support with load, memory, temperature, power
- **System Panel**: Uptime, process count, system information
- **Auto-refresh**: Updates every 2 seconds

### 2. Advanced Monitor
Ideal for long-term monitoring and analysis:
- **Everything from Enhanced Monitor**
- **Alert System**: Configurable thresholds for CPU, memory, GPU, disk
- **Process Monitoring**: Top CPU and memory consuming processes
- **Performance Trends**: Historical data with trend analysis
- **Alert Logging**: Persistent alert history with timestamps
- **Performance Reports**: Detailed JSON reports for analysis

### 3. GPU Monitor Pro ⭐ NEW!
Dedicated advanced GPU monitoring with comprehensive analytics:
- **Detailed GPU Metrics**: Clock speeds, utilization, encoder/decoder usage
- **Memory Analysis**: VRAM usage, memory processes, fragmentation tracking
- **Thermal Management**: Temperature monitoring, throttling detection
- **Power Efficiency**: Power draw, power limits, efficiency metrics
- **Process Tracking**: GPU processes with memory usage per process
- **Performance History**: 10-minute historical data with trend analysis
- **Multi-GPU Support**: Individual detailed monitoring for each GPU
- **Alert System**: GPU-specific alerts for temperature, memory, throttling
- **Performance Reports**: Comprehensive GPU performance analytics

### 4. System Overview
Quick status check for all system components:
- **Instant snapshot** of system health
- **Color-coded status** indicators
- **No continuous monitoring** - perfect for quick checks

## Alert Thresholds (Advanced Monitor)

Default alert thresholds can be customized in `advanced_monitor.py`:

```python
alert_thresholds = {
    'cpu_percent': 85.0,        # CPU usage percentage
    'memory_percent': 90.0,     # Memory usage percentage
    'gpu_percent': 90.0,        # GPU usage percentage
    'gpu_memory_percent': 95.0, # GPU memory percentage
    'gpu_temperature': 85.0,    # GPU temperature (°C)
    'disk_percent': 95.0        # Disk usage percentage
}
```

## Output Files

The advanced monitor generates several output files:
- `system_monitor.log`: Alert and system event log
- `performance_report.json`: Detailed performance statistics

## Key Features

### 🎨 Visual Features
- **Color-coded status indicators** (🔥 HIGH, ⚠️ MEDIUM, ✅ NORMAL)
- **Rich console output** with tables, panels, and progress bars
- **Real-time updates** with smooth refresh
- **Emoji indicators** for quick status recognition

### 📊 Monitoring Capabilities
- **Multi-GPU support** with individual GPU tracking
- **Per-core CPU monitoring** with detailed metrics
- **Memory usage tracking** including swap
- **Process monitoring** with top CPU/memory consumers
- **Temperature monitoring** for CPU and GPU
- **Power consumption tracking** (when available)

### 🚨 Alert System
- **Configurable thresholds** for all monitored components
- **Real-time alerts** with severity levels (WARNING, CRITICAL)
- **Persistent logging** with timestamps
- **Historical alert tracking**

### 📈 Performance Analysis
- **Trend analysis** with visual indicators
- **Historical data collection** with configurable retention
- **Performance reports** in JSON format
- **Statistical analysis** (average, max, min values)

## Requirements

- **Python 3.11+** (configured in pyproject.toml)
- **NVIDIA GPU** with drivers installed
- **Modern terminal** with color support
- **Unix-like system** (Linux, macOS)

## Dependencies

All dependencies are managed automatically with uv:
- `nvitop>=1.3.0` - Base NVIDIA monitoring
- `rich>=14.1.0` - Rich console output
- `psutil>=7.0.0` - System information
- `GPUtil>=1.4.0` - GPU utilities
- `numpy>=2.3.2` - Numerical operations
- `pandas>=2.3.2` - Data analysis
- `matplotlib>=3.10.6` - Plotting capabilities
- `seaborn>=0.13.2` - Statistical visualization

## Tips

1. **Use Enhanced Monitor** for daily monitoring with beautiful visuals
2. **Use Advanced Monitor** when you need alerts and historical tracking
3. **Run in tmux/screen** for persistent monitoring sessions
4. **Check performance reports** for long-term analysis
5. **Customize alert thresholds** based on your system's normal operation

## Troubleshooting

- **No GPUs detected**: Ensure NVIDIA drivers are installed
- **Permission errors**: Some system sensors may require elevated privileges
- **High CPU usage**: The monitors use minimal resources, but reduce update frequency if needed
- **Terminal compatibility**: Use a modern terminal with color and Unicode support

---

**Enjoy powerful, colorful, and insightful monitoring of your GPU and CPU systems!** 🚀