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

    h5_files = glob.glob(os.path.expanduser(args.creation_dataset_path) + '*.h5')
    dataset = ShhsDataset(h5_files)

    fs_channels, channels, _, annotation_labels = dataset.get_dataset_info()

    model = ModelCNN(num_classes=len(annotation_labels),
                      fs_channels=fs_channels
                      )

    routine = ModelTrainingRoutine(model, args)

    #routine.wandb_init(args)
    routine.run(dataset,
                annotation_labels,
                channels,
                args.num_epoch,
                args.batch_size,
                args.train_size
                )

if __name__ == '__main__':
    main()
