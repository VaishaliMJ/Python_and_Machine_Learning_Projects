"""-----------------------------------------------------------------------------------------------------
                        System Process Logger with Scheduling
                                Vaishali Jorwekar
--------------------------------------------------------------------------------------------------------
Problem statement:Design a Python Automation project to periodically log details of running processes(PID, name, user, memory usage) on the system. Each execution  generates a new log file with a timestamped filename for easy tracking
                -Implement process scanning using Psutil to extract process details
                -Automate log file creation with timestamps and proper formatting
                -Use the schedule library to run logging tasks periodically without manual intervention
                -Enhanced usability by allowing custom folder name and interval input
--------------------------------------------------------------------------------------------------------"""
########################################################################################################
#   Imports 
########################################################################################################
import logging,os,time
import argparse,sys
import datetime,schedule
import psutil,os,time,schedule

########################################################################################################
# Constants
########################################################################################################
BORDER="-"*60 

APPLICATION_LOG_FILE = f"Application_log"
PROCESS_LOG=f"Running_Process_log"

########################################################################################################
#   Function        :   getCurrFormattedTime
#   Input Params    :   None
#   Output Params   :   Current formatted time
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
########################################################################################################
def getCurrFormattedTime():
    currDateTime=datetime.datetime.now()
    currTime=currDateTime.strftime("%I:%M:%S %p")
    return currTime  
########################################################################################################
#   Function        :   formatTime
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
########################################################################################################
def formatTime():
    currTime=getCurrFormattedTime()
    currTime=currTime.replace(":","_")
    currTime=currTime.replace(" ","_")
    return currTime  
########################################################################################################    



###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   11 Nov 2025
############################################################################################
def ensure_dir(path:str):
    try:
        os.makedirs(path,exist_ok=True) 

    except OSError as e:
        print(f"An OS error occurred: {e}")
        exit()
    except Exception as e:
        print(f"An Exception occurred: {e}")
        exit()  
###########################################################################################
#   Function        :   ensure_logFile
#   Input Params    :   logDirectoryName(str)-directory Name
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   11 Nov 2025
############################################################################################
def ensure_logFile(logDirectoryName):
    try:
        logFileName = f"{APPLICATION_LOG_FILE}_{formatTime()}.log"

        ensure_dir(logDirectoryName)
        logFilePath=os.path.join(logDirectoryName,logFileName)
        logging.basicConfig(filename=logFilePath, level=logging.INFO,filemode="w",
                    format='%(asctime)s - %(levelname)s- %(funcName)s- %(message)s')
        
    except IOError as e:
        # Code to execute for other I/O errors
        print(f"An I/O error occurred: {e}")
        sys.exit()
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        sys.exit()
    return logFilePath  
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,interference baselines
#   Author          :   Vaishali M Jorwekar
#   Date            :   11 Nov 2025
############################################################################################
def parse_args():
    p=argparse.ArgumentParser(description="Directory File Cleaner Automation Script")
    p.add_argument("--LogDirectoryName",required=True,
                   type=str,help="Name of the directory to store Process Log details !!! ")
    p.add_argument("--duration",required=True,default=30,
                   type=int,help="Schedular duration")
    return p.parse_args()
#####################################################################################################
#   Function Name    :  initScript
#   Description      :  This function initialise the script
#   Input Params     :  Command line arguments as a input  
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  11 Nov 2025
#####################################################################################################
def initScript(args):
    logDirectoryName = args.LogDirectoryName

    duration=args.duration
    #   Create Log folder
    ensure_dir(logDirectoryName)
    #   Create application log file
    ensure_logFile(logDirectoryName)
    # Schedular duration
    runSchedular(duration,logDirectoryName)
#####################################################################################################
#   Function Name    :  runSchedular
#   Description      :  This function runs the schedular and finds duplicate files
#   Input Params     :  duration,logDirectoryName(Log Directory)
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  11 Nov 2025
#####################################################################################################
def runSchedular(duration,logDirectoryName):
    try:
        schedule.every(duration).minutes.do(processLog,logDirectoryName)
        while(True):
            schedule.run_pending()
            time.sleep(1)
    except ValueError as valErr:
        logging.error(f"Number expected...{valErr}")
    except Exception as excObj:
        logging.error(f"Exception occured {excObj}")    
#####################################################################################################
#   Function Name    :  processLog
#   Description      :  This function finds and logs currently running processes in a log file
#   Input Params     :  logDirectoryName(Log Directory)
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  11 Nov 2025
#####################################################################################################
def processLog(logDirectoryName):
    try:    
        logFileName = f"{PROCESS_LOG}_{formatTime()}.log"
    
        FileName=os.path.join(logDirectoryName,logFileName)

        fobj=open(FileName,"w")
        logging.info(f"Created Log file : {FileName}")
        fobj.write(BORDER)
        fobj.write("\n\t\tRunning Process Log\n")
        fobj.write("\t\tLog File is Created at:"+time.ctime()+"\n")
        fobj.write(BORDER+"\n")
        # Finds Running processes
        Data=processScan()

        for value in Data:
            fobj.write("%s \n"%value)
            fobj.write("\n")
            fobj.write(BORDER)
            fobj.write("\n")
        fobj.close()   
        logging.info(f"Successfully logged running process details in :{logFileName}")
    except  Exception as e:
        logging.error(f"Exception Occured {e}") 
#####################################################################################################
#   Function Name    :  processScan
#   Description      :  This function finds and logs currently running processes
#   Input Params     :  None
#   Output Params    :  Running process list
#   Author           :  Vaishali M Jorwekar
#   Date             :  11 Nov 2025
#####################################################################################################        
def processScan():
    listProcess=[]

    for proc in psutil.process_iter():  
        try:    
            info=proc.as_dict(attrs=['pid','name','username'])
            info['vms']=proc.memory_info().vms/(1024*1024)
            
            listProcess.append(info)
            
        except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):    
            pass
        except Exception as Err:
            logging.error(f"Exception occured displayCurrentProcesses().:{Err}")  
    return listProcess        
#####################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   11 Nov 2025
#####################################################################################################
def main():
    
    #ensure_logFile()
    try:
        args=parse_args()
        #initialise Script
        initScript(args)
    except Exception as Err:
        logging.error(f"Error while processing program {Err}")
########################################################################################################
#   Entry point of the program
########################################################################################################
if __name__=="__main__":
    main()
########################################################################################################
    
