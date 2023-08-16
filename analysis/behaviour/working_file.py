from create_dataframe import create_behavioural_dataframe


# inside the root path directory files should be stored like: participant_id/all the result files.txt

df = create_behavioural_dataframe(root_path='C:/Users/mariu/OneDrive/Desktop/behavioural_analysis/data/result_files')

df.to_excel('C:/Users/mariu/OneDrive/Desktop/behavioural_analysis/new_behavioural_dataframe.xlsx')
