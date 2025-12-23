## 📊 System Process Logger with Scheduling
An automated Python utility designed for system administrators and developers to monitor system health. The script periodically captures all active processes, logging critical data points like PID, Memory Usage, and User ownership.

## 🚀 Key Features

- **Resource Monitoring:** Captures PID, Process Name, Username, and Memory Usage.
- **Automated Scheduling:** Leverages the `schedule` library for hands-free, periodic logging.
- **Dynamic File Management:** Generates unique, timestamped log files for every session to ensure data integrity.
- **Cross-Platform:** Works on Windows, macOS, and Linux thanks to the `psutil` library.

## 🛠️ Installation
## 💻 Dependencies:
  Install the required Python packages before running the project

      pip install Psutil, OS,Schedule,Time
 ---     
### 🏃🏻‍♂️Running Script:
#### Command line Arguments 
  
    Argument 1 :  --LogDirectoryName (Process Log Directory Name)
  
    Argument 2 :  --duration ( Script duration in minutes)

#### Example:
    python SystemProcessLogger.py --LogDirectoryName "ProcessMonitorLogs" --duration 1
    
---
    
#### ✍️Author:
Vaishali M. Jorwekar<br>
Date :11 Nov 2025



