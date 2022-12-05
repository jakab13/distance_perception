import slab
import datetime

#creatin the storage file
def create_and_store_file(parent_folder, subject_folder, subject_id, trialsequence, group):
    file = slab.ResultsFile(subject=subject_folder, folder=parent_folder)
    subject_id = subject_id
    file.write(subject_id, tag='subject_ID')
    today = datetime.now()
    file.write(today.strftime('%Y/%m/%d'), tag='Date')
    file.write(today.strftime('%H:%M:%S'), tag='Time')
    file.write(group, tag='group')
    file.write(trialsequence, tag='Trial')
    return file