# Overview
**System Process Logger with Scheduling**


  **Description**: Designed a Python Automation project to periodically 
  log details of running processes(PID, name, user, memory usage) on the system. 

  Each execution  generates a new log file with a timestamped filename for easy tracking

    -Implemented process scanning using Psutil to extract process details
  
    -Automated log file creation with timestamps and proper formatting
  
    -Used the schedule library to run logging tasks periodically without manual intervention
  
    -Enhanced usability by allowing custom folder name and interval input

  ## Dependencies:
  Install the required Python packages before running the project

      pip install Psutil, OS,Schedule,Time
      
  ### Running Script:
  #### Command line Arguments 
  
    Argument 1 :  --LogDirectoryName (Process Log Directory Name)
  
    Argument 2 :  --duration ( Script duration in minutes)

  #### Example:
    python SystemProcessLogger.py --LogDirectoryName "ProcessMonitorLogs" --duration 1

  #### Author:

  #### Vaishali M. Jorwekar

  #### Date :11 Nov 2025



