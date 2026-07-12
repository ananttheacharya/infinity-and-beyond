import subprocess
import sys
import time

def run_command(command, description):
    print(f"\n==================================================")
    print(f" [ORCHESTRATOR] {description}")
    print(f" COMMAND: {command}")
    print(f"==================================================\n")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)
        
    print(f"\n[SUCCESS] Completed in {end_time - start_time:.2f} seconds.")

def main():
    print("""
    ██████╗ ██╗██████╗ ████████╗
    ██╔══██╗██║██╔══██╗╚══██╔══╝
    ██████╔╝██║██║  ██║   ██║   
    ██╔═══╝ ██║██║  ██║   ██║   
    ██║     ██║██████╔╝   ██║   
    ╚═╝     ╚═╝╚═════╝    ╚═╝   
    PHYSICS-INFORMED DIGITAL TWIN ORCHESTRATOR
    """)
    
    print("Welcome to the Master Orchestrator.")
    print("This script will execute the entire machine learning pipeline.")
    
    # 1. Train the multi-task PINN on the real dataset
    run_command("python src/training/train_pinn.py", "TRAINING PHYSICS-INFORMED NEURAL NETWORK")
    
    print("\n==================================================")
    print(" [ORCHESTRATOR] PIPELINE COMPLETE")
    print(" The PINN model has been successfully trained and saved.")
    print(" You can now run 'python src/evaluation/telemetry_streamer.py' to launch the live dashboard stream.")
    print("==================================================")

if __name__ == "__main__":
    main()
