import hydra
from src.seeds import DeterministicSeed
from src.shhsdataset import ShhsDataset
from src.model import ModelCNN
from src.trainroutine import ModelTrainingRoutine
from src.camcaluculator import CamCalculator
import os
import glob
import torch


@hydra.main(config_path="config", config_name="parameters")
def main(args):
    DeterministicSeed(seed_num=args.seed)

    h5_files = glob.glob(os.path.expanduser(args.creation_dataset_path) + '*.h5')
    dataset = ShhsDataset(h5_files)

    fs_channels, channel_labels, _, annotation_labels = dataset.get_dataset_info()

    model = ModelCNN(num_classes=len(annotation_labels),
                      fs_channels=fs_channels,
                      # [batch_size, num_channels, features, sequence]
                      input_shape=[args.batch_size, len(fs_channels), 1, int(max(fs_channels)*args.duration)]
                      )

    #cam_calc = CamCalculator('/home/yusuke.sakai/workspace/nsrr/outputs/2024-02-20/23-43-21/model_e6.pth',
    cam_calc = CamCalculator('/home/yusuke.sakai/workspace/nsrr/outputs/2024-02-21/03-22-06/model_e36.pth',
                             args.gpu, annotation_labels, fs_channels, channel_labels, 6)
    train_dataset, test_dataset = dataset.split()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )
    #cam_calc.plot_cam(test_loader)
    cam_calc.calc_sim_result(test_loader)

if __name__ == '__main__':
    main()
