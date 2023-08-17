import sys
path = "/Users/jakabpilaszanovich/Documents/GitHub/distance_perception/analysis/EEG/eeg_tools/src/eeg_tools"
sys.path.append(path)
import preprocessing as pre
import pathlib
import utils
import analysis
import settings


# TODO: position _fig_folder somewhere else and make it more dynamic.
for id in settings.ids[:1]:  # loop through all ids
    is_preprocessed = False  # set.check_id(id, settings.root_dir)  # check if subject is already preprocessed
    if is_preprocessed is False:  # if not preprocessed, run loop
        print(f"START PREPROCESSING PARTICIPANT {settings.ids.index(id)} ({id})")
        # utils.generate_folders(root_dir=settings.data_dir, id=id)  # make data folders
        raw_folder = pathlib.Path(f"/Users/jakabpilaszanovich/Documents/GitHub/distance_perception/analysis/EEG/data/pinknoise/{id}/raw")
        fig_folder = pathlib.Path(f"/Users/jakabpilaszanovich/Documents/GitHub/distance_perception/analysis/EEG/data/pinknoise/{id}/figures")
        raw = pre.make_raw(settings.header_files, id, fig_folder, settings.mapping, settings.montage)  # make raw object
        utils.save_object(raw, settings.data_dir, id)  # save raw
        # raw = pre.read_raw(raw_folder, id)
        epochs = pre.run_pipeline(raw=raw, fig_folder=fig_folder, config=settings.cfg, exclude_event_id=7)  # run pipeline on raw
        del raw  # save working memory
        utils.save_object(epochs, settings.data_dir, id)  # save epochs
        evokeds = pre.make_evokeds(epochs, baseline=(None, 0))  # make evokeds, optionally apply baseline
        utils.save_object(evokeds, settings.data_dir, id)  # save evokeds
        del epochs, evokeds  # save working memory
        with open(settings.data_dir / id / f"config_{id}.txt", "w") as file:
            file.write(str(settings.cfg))  # save configuration file to keep track of the parameters per subject
        print(f"PARTICIPANT {settings.ids.index(id)} ({id}) SUCCESSFULLY PREPROCESSED!")
    else:
        continue

# After running the pipeline over all participants, do:
# analysis.quality_check(settings.ids, out_folder=settings.root_dir/"qc")

