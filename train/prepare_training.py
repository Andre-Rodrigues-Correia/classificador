from utils.utils import Utils


def prepare_models():
    Utils.format_file_from_train('PTBR', 'datasets/mid_dataset_PTBR.txt')
    Utils.format_file_from_train("PTPT", "datasets/mid_dataset_PTPT.txt")

