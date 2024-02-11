import hydra
from src.shhsdataload import ShhsDataLoader
import os

@hydra.main(config_path="config", config_name="parameters")
def main(args):
    shhs = ShhsDataLoader(args.annotation_labels,
                          base_path=args.original_dataset_path,
                          datasets=args.datasets,
                          output_csv=args.output_shhs_datainfo_csv,
                          verbose=args.verbose,
                          interp=args.interp)

    if args.output_shhs_datainfo_csv is True:
        shhs.generate_annot_counts_csv()

    shhs.create_target_fs_dataset_h5(args.channel_labels,
                                     fs_channels=args.fs_channels,
                                     target_fs=args.target_fs,
                                     creation_dataset_path=os.path.expanduser(args.creation_dataset_path),
                                     debug_plots_interval=args.debug_plots_interval)

if __name__ == '__main__':
    main()
