"""-----------------------------------------------------------------------------------------------------
                    Duplicate file cleaner
                        Vaishali Jorwekar
--------------------------------------------------------------------------------------------------------
Problem statement:Develope a Python-based automation script to detect and
                  delete duplicate files from a directory periodically.
                  The system automatically generates log files of operations 
                  and shares them via Email for audit purposes.
                    -Implement checksum-based duplicate detection using hash lib(MD5)
                    -Automate log generation with timestamp for every execution
                    -Used schedule library for periodic file cleanup
                    -Integrate email automation (smtplib)to send logs automatically
--------------------------------------------------------------------------------------------------------"""
########################################################################################################
#   Imports 
########################################################################################################
import logging,os,time
import argparse,sys
import datetime,schedule
import hashlib
from email_validator import validate_email, EmailNotValidError
from dotenv import load_dotenv 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
########################################################################################################
# Constants
########################################################################################################
BORDER="-"*60
 
#   Log directory
LOG_DIRECTORY="LOG_FOLDER"
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
#   Configure logging
SCRIPT_START_TIME=getCurrFormattedTime()
DELETED_FILES_LOG=f"Deleted_File_Log"
LOG_FILE = f"Program_Log_{formatTime()}.log"

########################################################################################################


###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
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
#   Function        :   DeletedFiles_logFile
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
############################################################################################
def DeletedFiles_logFile():
    try:
        ensure_dir(LOG_DIRECTORY)
        deletedFile_log=f"{DELETED_FILES_LOG}_{formatTime()}.log"
        deleted_logFilePath=os.path.join(LOG_DIRECTORY,deletedFile_log)
        deletedFile_Logger=logging.getLogger('info_logger')
        deletedFile_Logger.setLevel(logging.INFO)

        deleted_handler=logging.FileHandler(deleted_logFilePath)
        deleted_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s- %(funcName)s- %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S"))
        
        deletedFile_Logger.addHandler(deleted_handler)
    except IOError as e:
        # Code to execute for other I/O errors
        logging.error(f"An I/O error occurred: {e}")
        sys.exit()
    except Exception as e:
        # Catch any other unexpected exceptions
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit()
    return deletedFile_Logger,deleted_logFilePath  
###########################################################################################
#   Function        :   ensure_logFile
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
############################################################################################
def ensure_logFile():
    try:
        ensure_dir(LOG_DIRECTORY)
        logFilePath=os.path.join(LOG_DIRECTORY,LOG_FILE)
        logging.basicConfig(filename=logFilePath, 
                            level=logging.INFO,
                            filemode="w",
                            format='%(asctime)s - %(levelname)s- %(funcName)s- %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
        
    except IOError as e:
        # Code to execute for other I/O errors
        print(f"An I/O error occurred: {e}")
        sys.exit()
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        sys.exit()
    return logFilePath
#####################################################################################################
#   Function Name    :  checkDirectoryExsist
#   Description      :  This function validates the directory
#   Input Params     :  Directory Name
#   Author           :  Vaishali M Jorwekar
#   Date             :  9 Nov 2025
#####################################################################################################
def validateDirectory(directoryName):      
      # Return whether a path is absolute
      flag=os.path.isabs(directoryName)
      if(flag==False):
        #get absolute directory path
        directoryName=os.path.abspath(directoryName)
      #Check if directory path exists
      flag=os.path.exists(directoryName)
      if(flag==False):
          logging.error(f"Invalid directory path :'{directoryName}' ")
          sys.exit() 
      #Check if given name is directory  
      flag=os.path.isdir(directoryName)
      if(flag==False):
        logging.error(f"Input name '{directoryName}' is not name of directory.")
        sys.exit()  
      return directoryName     
#####################################################################################################
#   Function Name    :  initScript
#   Description      :  This function initialise the script
#   Input Params     :  Command line arguments as a input  
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  9 Nov 2025
#####################################################################################################
def initScript(args):
    directoryName = args.directoryName
    emailId=args.emailId
    duration=args.duration
    logging.info(BORDER)
    logging.info(f"Duplicate file cleanup for directory : {directoryName}")
    logging.info(BORDER)  
    # Check if given directory exsist 
    directoryName=validateDirectory(directoryName) 
    #Check if valid email Id
    validateEmail(emailId)
    # Schedular duration
    runSchedular(duration,directoryName,emailId)
#####################################################################################################
#   Function Name    :  runSchedular
#   Description      :  This function runs the schedular and finds duplicate files
#   Input Params     :  duration,directoryName(Input Directory),emailId
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  9 Nov 2025
#####################################################################################################
def runSchedular(duration,directoryName,emailId):
    try:
        schedule.every(duration).minutes.do(findAndDeleteDuplicateFiles,directoryName,emailId)
        while(True):
            schedule.run_pending()
            time.sleep(1)
    except ValueError as valErr:
        logging.error(f"Number expected...{valErr}")
    except Exception as excObj:
        logging.error(f"Exception occured {excObj}")
#####################################################################################################
#   Function Name    :  findAndDeleteDuplicateFiles
#   Description      :  This function find and deleted duplicate files and sends an email
#   Input Params     :  directoryName(Input Directory) 
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  9 Nov 2025
#####################################################################################################
def findAndDeleteDuplicateFiles(directoryName,emailId):
    try:
        deletedFile_Logger,deleted_logFilePath=DeletedFiles_logFile()
        #   Directory travesal and duplicate logic
        DuplicateDict,totalScannedFile=directoryTraversal(directoryName)
        #   Remove duplicate files
        totalDuplicateDeleted=removeDuplicateFiles(DuplicateDict,deletedFile_Logger,directoryName)  
        #   Send an Email 
        sendEmail(totalScannedFile,totalDuplicateDeleted,emailId,deleted_logFilePath)
    except  Exception as e:
        logging.error(f"Exception Occured {e}") 
#####################################################################################################
#   Function Name    :  sendEmail
#   Description      :  This function find and deleted duplicate files and sends an email
#   Input Params     :  totalScannedFile,totalDuplicateDeleted,receiverEmailId,deleted_logFilePath
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  9 Nov 2025
#####################################################################################################        
def sendEmail(totalScannedFile,totalDuplicateDeleted,receiverEmailId,deleted_logFilePath):
    try:  
        # loading variables from .env file
        load_dotenv()       
        outputFileName=deleted_logFilePath
        # message to be sent
        subject = "Process log file created..."

        body = "Hi,\nDeleted duplicate files log.\n" \
                 f"Execution start time : {SCRIPT_START_TIME}\n" \
                 f"Total number of files scanned:{totalScannedFile}\n" \
                 f"Total number of Duplicate files found:{totalDuplicateDeleted}"
        

    # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = os.getenv("SENDER_EMAIL_ID")
        message["To"] = receiverEmailId
        message["Subject"] = subject

        # Add body to email
        message.attach(MIMEText(body, "plain"))


        # Open the file in binary mode
        with open(outputFileName, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
           # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

         # Add header as key/value pair to attachment part
        part.add_header("Content-Disposition", f"attachment; filename= {outputFileName}")

        # Add attachment to message
        message.attach(part)
        # Send the email
        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("PORT"))) as server:
            server.starttls()
            server.login(os.getenv("LOGIN"), os.getenv("SENDER_PASSWORD"))
            server.sendmail(os.getenv("SENDER_EMAIL_ID"), receiverEmailId, message.as_string())

        logging.info(f"Email sent successfully to '{receiverEmailId}'")
        
    except Exception as errObj:
         logging.error("\nExeception in sendEmail() method:",errObj)            
#####################################################################################################
#   Function Name    :  directoryTraversal
#   Description      :  This function finds and delete duplicate files
#   Input Params     :  directoryName(Input Directory) 
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  9 Nov 2025
#####################################################################################################
def directoryTraversal(directoryName):
    try:
        totalFileCount=0
        DuplicateDict={}
        for folderName, SubFolderNames, fileNames in os.walk(directoryName):
            
            for fileName in fileNames:
                totalFileCount=totalFileCount+1
                fileName=os.path.join(folderName,fileName)
                checkSum=findCheckSum(fileName)
                if checkSum in DuplicateDict:
                    DuplicateDict[checkSum].append(fileName)
                else:
                    DuplicateDict[checkSum]=[fileName]       
    except Exception as e:
        logging.error(f"Exception Occured {e}") 
    return DuplicateDict,totalFileCount

#####################################################################################################
#   Function Name    :  removeDuplicateFiles
#   Description      :  This function removes the duplicate files
#   Input Params     :  Duplicate files dictionary,DeletedFiles_logFile,directoryName
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  9 Nov 2025
#####################################################################################################
def removeDuplicateFiles(DuplicateDict,DeletedFiles_logFile,directoryName):
    try:
        duplicateFileList=list(filter(lambda fileList : len(fileList)>1,DuplicateDict.values()))
        totalDuplicateDeleted=0
        for fileNamesList in duplicateFileList:
            DeletedFiles_logFile.info(f"Duplicate Files:{fileNamesList}")
            count=0
            for fileName in fileNamesList:
                DeletedFiles_logFile.info(f"File Name:\t{fileName}") 
                count=count+1
                if(count>1):
                    os.remove(fileName)
                    totalDuplicateDeleted=totalDuplicateDeleted+1
                    DeletedFiles_logFile.info(f"Deleted File :{fileName}") 
        if totalDuplicateDeleted > 0:            
            DeletedFiles_logFile.info(BORDER)
            DeletedFiles_logFile.info(f"Total deleted duplicate files :{totalDuplicateDeleted}")
            DeletedFiles_logFile.info(BORDER)
        else:
            DeletedFiles_logFile.info(BORDER)
            DeletedFiles_logFile.info(f"No duplicate files found in folder  :{directoryName}")
            DeletedFiles_logFile.info(BORDER)   

    except Exception as e:
        logging.error(f"Exception Occured {e}")  
    return totalDuplicateDeleted          
#####################################################################################################
#   Function Name    :  findCheckSum
#   Description      :  This function calculates the checksum of the file
#   Input Params     :  File Name 
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  9 Nov 2025
#####################################################################################################
def findCheckSum(fileName):
    try:
        #Define buffer size
        BlockSize=1024
        #   Open file in binary mode
        fObj= open(fileName,"rb")
        #Use hashlib md5 algorithm to find checksum of file
        hobj=hashlib.md5()
        bufferedData=fObj.read(BlockSize)

        while(len(bufferedData)>0):
            hobj.update(bufferedData)
            bufferedData=fObj.read(BlockSize)
        fObj.close()   
          
    except Exception as e:
        logging.error(f"Exception Occured  while calculating checksum of {fileName} : {e}") 
    return hobj.hexdigest()
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,interference baselines
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
############################################################################################
def parse_args():
    p=argparse.ArgumentParser(description="Directory File Cleaner Automation Script")
    p.add_argument("--directoryName",required=True,
                   type=str,help="Name of the directory to clean duplicate files !!! ")
    p.add_argument("--duration",required=True,default=30,
                   type=int,help="Schedular duration")
    p.add_argument("--emailId",required=True,
                   type=str,help="Receiver Email Id")
    return p.parse_args()
###########################################################################################
#   Function        :   validateEmail
#   Input Params    :   Receiver Email Id
#   Output Params   :   -
#   Description     :   Validates EMail Id
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
############################################################################################
def validateEmail(email):
    try:
      # validate and get info
        validate = validate_email(email) 
        # replace with normalized form
        email = validate.email 
        
    except EmailNotValidError as e:
        # email is not valid, exception message is human-readable
        logging.error(f"Email Id is Not valid :{email} : {str(e)}")
        exit()
#####################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
#####################################################################################################
def main():
    ensure_dir(LOG_DIRECTORY)
    ensure_logFile()
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
    