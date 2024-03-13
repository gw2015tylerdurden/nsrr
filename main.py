import hydra
from src.seeds import DeterministicSeed
from src.shhsdataset import ShhsDataset
from src.model import ModelCNN
from src.trainroutine import ModelTrainingRoutine
import os
import glob


@hydra.main(config_path="config", config_name="parameters")
def main(args):
    DeterministicSeed(seed_num=args.seed)

    pop_channels = []
    h5_files = glob.glob(os.path.expanduser(args.creation_dataset_path) + '*.h5')

    for i in range(len(args.fs_channels)):
        # mkdir loop_${i}; mv loop_${i}
        directory = f"loop_{i}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)

        dataset = ShhsDataset(h5_files, pop_channels, args.replace_noise_channel_labels)
        fs_channels, channels, _, annotation_labels = dataset.get_dataset_info()

        model = ModelCNN(num_classes=len(annotation_labels),
                         fs_channels=fs_channels
                         )

        routine = ModelTrainingRoutine(model, fs_channels, channels, annotation_labels, args)
        routine.wandb_init(args)

        s_min_idx, s_max_idx = routine.run(dataset,
                                           args.num_epoch,
                                           args.batch_size,
                                           args.train_size
                                           )
        pop_channels.append(s_min_idx)

        # cd ..
        os.chdir("..")

if __name__ == '__main__':
    main()
