import logging
import os
from datetime import datetime

LOG_FILE=F"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #Create a timestamped log filename
logs_path=os.path.join(os.getcwd(),"logs")#create 'logs' directory
os.makedirs(logs_path,exist_ok=True)



LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)
#combines folder path and the file name
#project/logs/12_19_2025_02_45_10.log

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= '%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)

if __name__=="__main__":
    logging.info("Logging has started")