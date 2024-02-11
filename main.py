import hydra
from src.seeds import DeterministicSeed
from src.shhsdataset import ShhsDataset
from src.trainroutine import ModelTrainingRoutine
import os
import glob


@hydra.main(config_path="config", config_name="parameters")
def main(args):
    DeterministicSeed(seed_num=args.seed)

    pop_channels = []
    h5_files = glob.glob(os.path.expanduser(args.creation_dataset_path) + '*.h5')

    loop_count = 0
    while True:
        # mkdir loop_${i}; mv loop_${i}
        directory = f"loop_{loop_count}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)

        dataset = ShhsDataset(h5_files, pop_channels, args.add_noise_channel_fs)
        fs_channels, channels, _, annotation_labels = dataset.get_dataset_info()

        routine = ModelTrainingRoutine(fs_channels, channels, annotation_labels, args)

        s_min_idx, s_max_idx = routine.run(dataset,
                                           args,
                                           args.num_epoch,
                                           args.batch_size,
                                           loop_count)
        pop_channels.append(channels[s_min_idx])

        if len(fs_channels) == 1:
            # Can't pop any more signals
            break
        else:
            loop_count += 1

        # cd ..
        os.chdir("..")

    with open("pop_channels.txt", "w") as file:
        file.write(pop_channels)

if __name__ == '__main__':
    main()
