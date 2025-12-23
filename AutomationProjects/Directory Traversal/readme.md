## 📂 Duplicate File Cleaner 🧹

An automated Python utility that periodically cleans directories of duplicate files using **MD5 checksum verification** and sends detailed audit logs via email.

---

### ✨ Features
- **Smart Detection:** Uses MD5 hashing to find truly identical files, regardless of filename.
- **Scheduled Automation:** Runs cleanup tasks at precise intervals (e.g., every 60 minutes).
- **Automated Auditing:** Generates timestamped logs and emails them automatically to administrators.
- **Secure Config:** Protects SMTP credentials using `.env` environment variables.

## 🛠️ Installation
### Dependencies 
1. #### Python Libraries Installation
    Install the required Python packages before running the project
      pip install time os sys schedule hashlib smtplib dotenv email argparse logging email_validator
2. #### Configure Environment:
   Create a .env file in the root directory:<br>
    Update the values in .env file<br>
  *       EMAIL_USER="your-email@gmail.com"
  *       EMAIL_PASS="your-app-specific-password"

---

 ### 🚀 Usage
Run the script from the command line with your target parameters:
python3 DirectoryDuplicateFileCleaner.py --directoryName "/path/to/folder" --duration 30 --emailId "receiver@example.com"

 #### Command line Arguments
      Argument 1 : --directoryName(Directory Name to be cleaned for duplicate files)
  
      Argument 2 : --duration(In Minutes)  
  
      Argument 3 : --emailId(Receiver name to send the process log)

  ---
 #### 🛡️ Safety Disclaimer
**Warning**: This script permanently deletes files. It is recommended to test on a non-critical directory before full deployment.

### Author
Vaishali M. Jorwekar<br>
Date	:11 Nov 2025

 
