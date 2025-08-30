#!/usr/bin/env python3
"""
GPU Monitor Launcher
Easy-to-use launcher for different monitoring modes
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

def show_menu():
    """Show interactive menu for monitor selection"""
    console = Console()
    console.clear()
    
    # Create title
    title = Text("🚀 Enhanced GPU & CPU Monitor", style="bold green")
    console.print(Panel(title, style="bold blue"))
    
    # Create options table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan", width=10)
    table.add_column("Monitor Type", style="green", width=25)
    table.add_column("Description", style="white")
    
    table.add_row("1", "🎯 Original nvitop", "Basic NVIDIA GPU monitoring")
    table.add_row("2", "🌈 Enhanced Monitor", "Colorful real-time CPU/GPU dashboard")
    table.add_row("3", "🔍 Advanced Monitor", "Full monitoring with alerts & trends")
    table.add_row("4", "🎮 GPU Monitor Pro", "Dedicated advanced GPU analytics")
    table.add_row("5", "🔥 Kernel Monitor", "CUDA kernel & operator-level tracking")
    table.add_row("6", "🔧 Integrated Monitor", "All NVIDIA tools unified")
    table.add_row("7", "📊 System Overview", "Quick system status check")
    table.add_row("q", "❌ Quit", "Exit the monitor")
    
    console.print(table)
    console.print()
    
    while True:
        choice = console.input("[bold cyan]Select monitor mode (1-7, q): [/bold cyan]").strip().lower()
        
        if choice == '1':
            console.print("[green]Starting original nvitop...[/green]")
            import os
            os.system("uv run nvitop")
            break
        elif choice == '2':
            console.print("[green]Starting enhanced monitor...[/green]")
            from enhanced_monitor import main as enhanced_main
            enhanced_main()
            break
        elif choice == '3':
            console.print("[green]Starting advanced monitor...[/green]")
            from advanced_monitor import main as advanced_main
            advanced_main()
            break
        elif choice == '4':
            console.print("[green]Starting GPU Monitor Pro...[/green]")
            from gpu_monitor_advanced import main as gpu_main
            gpu_main()
            break
        elif choice == '5':
            console.print("[green]Starting Kernel Monitor...[/green]")
            from gpu_kernel_monitor import main as kernel_main
            kernel_main()
            break
        elif choice == '6':
            console.print("[green]Starting Integrated Monitor...[/green]")
            from integrated_gpu_monitor import main as integrated_main
            integrated_main()
            break
        elif choice == '7':
            show_system_overview()
            break
        elif choice == 'q':
            console.print("[yellow]Goodbye![/yellow]")
            break
        else:
            console.print("[red]Invalid choice. Please select 1-7 or q.[/red]")

def show_system_overview():
    """Show quick system overview"""
    console = Console()
    import psutil
    import GPUtil
    from datetime import datetime
    
    console.clear()
    console.print(Panel(Text("📊 System Overview", style="bold blue"), style="blue"))
    
    # System info
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan", width=15)
    table.add_column("Status", style="green", width=20)
    table.add_column("Details", style="white")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_status = "🔥 HIGH" if cpu_percent > 80 else "⚠️ MEDIUM" if cpu_percent > 60 else "✅ NORMAL"
    cpu_color = "red" if cpu_percent > 80 else "yellow" if cpu_percent > 60 else "green"
    table.add_row(
        "CPU", 
        f"[{cpu_color}]{cpu_status}[/{cpu_color}]",
        f"{cpu_percent:.1f}% usage, {psutil.cpu_count()} cores"
    )
    
    # Memory
    memory = psutil.virtual_memory()
    mem_status = "🔥 HIGH" if memory.percent > 85 else "⚠️ MEDIUM" if memory.percent > 70 else "✅ NORMAL"
    mem_color = "red" if memory.percent > 85 else "yellow" if memory.percent > 70 else "green"
    table.add_row(
        "Memory",
        f"[{mem_color}]{mem_status}[/{mem_color}]",
        f"{memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)"
    )
    
    # GPU
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                gpu_load = gpu.load * 100
                gpu_status = "🔥 HIGH" if gpu_load > 80 else "⚠️ MEDIUM" if gpu_load > 60 else "✅ NORMAL"
                gpu_color = "red" if gpu_load > 80 else "yellow" if gpu_load > 60 else "green"
                table.add_row(
                    f"GPU {i}",
                    f"[{gpu_color}]{gpu_status}[/{gpu_color}]",
                    f"{gpu.name} - {gpu_load:.1f}% load, {gpu.temperature}°C"
                )
        else:
            table.add_row("GPU", "[dim]Not detected[/dim]", "No NVIDIA GPUs found")
    except Exception:
        table.add_row("GPU", "[red]Error[/red]", "Unable to detect GPUs")
    
    # Disk
    disk_usage = psutil.disk_usage('/')
    disk_percent = (disk_usage.used / disk_usage.total) * 100
    disk_status = "🔥 FULL" if disk_percent > 90 else "⚠️ HIGH" if disk_percent > 80 else "✅ NORMAL"
    disk_color = "red" if disk_percent > 90 else "yellow" if disk_percent > 80 else "green"
    table.add_row(
        "Disk",
        f"[{disk_color}]{disk_status}[/{disk_color}]",
        f"{disk_percent:.1f}% used ({disk_usage.used/1024**3:.0f}GB/{disk_usage.total/1024**3:.0f}GB)"
    )
    
    console.print(table)
    console.print()
    console.print(f"[dim]System overview generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    console.print()
    console.input("[cyan]Press Enter to return to menu...[/cyan]")
    show_menu()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced GPU & CPU Monitor')
    parser.add_argument('mode', nargs='?', choices=['enhanced', 'advanced', 'gpu', 'kernel', 'integrated', 'overview', 'nvitop'], 
                       help='Monitor mode to run directly')
    parser.add_argument('--no-menu', action='store_true', help='Skip interactive menu')
    
    args = parser.parse_args()
    
    if args.mode:
        console = Console()
        if args.mode == 'enhanced':
            console.print("[green]Starting enhanced monitor...[/green]")
            from enhanced_monitor import main as enhanced_main
            enhanced_main()
        elif args.mode == 'advanced':
            console.print("[green]Starting advanced monitor...[/green]")
            from advanced_monitor import main as advanced_main
            advanced_main()
        elif args.mode == 'gpu':
            console.print("[green]Starting GPU Monitor Pro...[/green]")
            from gpu_monitor_advanced import main as gpu_main
            gpu_main()
        elif args.mode == 'kernel':
            console.print("[green]Starting Kernel Monitor...[/green]")
            from gpu_kernel_monitor import main as kernel_main
            kernel_main()
        elif args.mode == 'integrated':
            console.print("[green]Starting Integrated Monitor...[/green]")
            from integrated_gpu_monitor import main as integrated_main
            integrated_main()
        elif args.mode == 'overview':
            show_system_overview()
        elif args.mode == 'nvitop':
            import os
            os.system("uv run nvitop")
    else:
        show_menu()

if __name__ == "__main__":
    main()