# Overview
       Developed a Python-based automation script to detect and deleted duplicate files from a directory periodically
       The system automatically generates log files of operations and shares them via Email for audit purposes.
        -Implemented checksum-based duplicate detection using hash lib(MD5)
        -Automated log generation with timestamp for every execution
        -Used schedule library for periodic file cleanup
        -Integrated email automation (smtplib)to send logs automatically

 ## Dependencies 
    Install the required Python packages before running the project
   
     _pip install time os sys schedule hashlib smtplib dotenv email argparse logging email_validator_
 
 ### Running Script
 #### Command line Arguments
      Argument 1 : --directoryName(Directory Name to be cleaned for duplicate files)
  
      Argument 2 : --duration(In Minutes)  
  
      Argument 3 : --emailId(Receiver name to send the process log)

  #### Example :

      python3 DirectoryDuplicateFileCleaner.py --directoryName "Folder/Directory Path" --duration "Duration In Mins" --emailId "Receiver Email Id"

### Author
**Vaishali M. Jorwekar**

 **Date	:11 Nov 2025**

 
