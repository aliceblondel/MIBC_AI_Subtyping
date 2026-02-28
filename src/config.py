import argparse
import yaml
from argparse import Namespace

def load_yaml_config(config_path: str) -> list[str]:
    """Load a YAML file and convert it into arguments for argparse."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if 'DATA_FOLDER' in config:
        DATA_FOLDER = config.get("DATA_FOLDER", ".")
        for key, value in config.items():
            if isinstance(value, str):
                config[key] = value.replace("__DATA_FOLDER__", DATA_FOLDER)
        del config['DATA_FOLDER']


    # Convert the arguments into a list for argparse
    arg_list = []
    for key, value in config.items():
        if isinstance(value, list):  # For multi-value arguments
            arg_list.append(f"--{key}")
            arg_list.extend(map(str, value))
        else:
            arg_list.append(f"--{key}")
            arg_list.append(str(value))

    return arg_list

def save_args(args: Namespace, save_path: str) -> None:
    """Serialize a Namespace to a YAML file."""
    with open(save_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def get_model_args(config_path: str | None = None) -> Namespace:
    """Load only the model architecture parameters from a YAML file."""
    parser = argparse.ArgumentParser(add_help=False)

    # Dimensions
    parser.add_argument("--feature_depth", type=int, default=1536)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_genes", type=int, default=848)
    parser.add_argument("--ensembl_gene_ids", type=str, nargs='+', default=None)
    parser.add_argument("--hgnc_symbols", type=str, nargs='+', default=None)

    # Predictor
    parser.add_argument("--predictor_hidden_dims", type=int, nargs='+', default=[512, 512])
    parser.add_argument("--classifier_hidden_dims", type=int, nargs='+', default=[256])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--atn_dim", type=int, default=256)
    parser.add_argument("--classif_on_gene_exp", type=int, default=0)

    # Aggregator
    parser.add_argument("--pooling_fct", type=str, default="mean")
    parser.add_argument("--instance_based", type=int, default=0)

    # Encoder
    parser.add_argument("--encoder_type", type=str, default="identity")
    parser.add_argument("--nb_tiles", type=int, default=200)
    parser.add_argument("--encoder_num_heads", type=int, default=4)
    parser.add_argument("--encoder_num_layers", type=int, default=1)
    parser.add_argument("--encoder_ff_dim", type=int, default=2048)
    parser.add_argument("--encoder_dropout", type=float, default=0.3)

    arg_list = load_yaml_config(config_path) if config_path else []
    return parser.parse_known_args(arg_list)[0]

def get_mibc_model_args(config_path: str | None = None) -> Namespace:
    """Load only the model architecture parameters from a YAML file."""
    parser = argparse.ArgumentParser(add_help=False)

    # Dimensions
    parser.add_argument("--feature_depth", type=int, default=1536)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--classifier_hidden_dims", type=int, nargs='+', default=[256])

    arg_list = load_yaml_config(config_path) if config_path else []
    return parser.parse_known_args(arg_list)[0]

def get_predict_args(arg_list: list[str] | None = None, config_path: str | None = None) -> Namespace:
    """Parse arguments for the prediction pipeline (slide-level inference)."""
    parser = argparse.ArgumentParser(description="Arguments for slide-level evaluation")

    parser.add_argument("--patient_id_col", type=str, required=True)
    parser.add_argument("--slide_id_col", type=str, required=True)
    parser.add_argument("--aggregate_by_patient", type=int, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--emb_folder", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--compute_metrics", type=int, default=0)
    parser.add_argument("--adata_gt_path", type=str, default=None)
    parser.add_argument("--use_mibc_detect", type=int, default=1)
    parser.add_argument("--use_learnt_classifier", type=int, default=1)

    if config_path:
        arg_list = load_yaml_config(config_path)

    return parser.parse_args(arg_list)


def get_encode_args(arg_list: list[str] | None = None, config_path: str | None = None) -> Namespace:
    """Parse arguments for the tile encoding pipeline (H-Optimus-1)."""
    parser = argparse.ArgumentParser(description='Encode WSI tiles with H-optimus-1')

    parser.add_argument('--encode_by', type=str, default='slide_id')
    parser.add_argument('--slide_id_col', type=str, default='slide_id')
    parser.add_argument('--patient_id_col', type=str, default='patient_id')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--emb_folder', type=str, required=True)
    parser.add_argument('--slide_folder', type=str, required=True)
    parser.add_argument('--annotation_folder', type=str, default=None)
    parser.add_argument('--segmentation_folder', type=str, default=None)
    parser.add_argument('--final_tile_size', type=int, default=224)
    parser.add_argument('--magnification_tile', type=float, default=20)
    parser.add_argument('--save_images', type=int, default=0)
    parser.add_argument('--mask_tolerance', type=float, default=0.9)
    parser.add_argument('--hes2he', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)

    if config_path:
        arg_list = load_yaml_config(config_path)

    return parser.parse_args(arg_list)


def get_maps_args(arg_list: list[str] | None = None, config_path: str | None = None) -> Namespace:
    """Handle arguments coming either from a list (arg_list) or from a YAML file (config_path)."""
    parser = argparse.ArgumentParser(description="Arguments for generating maps")

    # Paths
    parser.add_argument("--slide_csv", type=str, required=True)
    parser.add_argument("--slide_emb_folder", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    # Columns
    parser.add_argument("--patient_id_col", type=str, required=True)
    parser.add_argument("--slide_id_col", type=str, required=True)
    # Flags
    parser.add_argument("--use_learnt_classifier", type=int, default=1)
    parser.add_argument("--display_gene_exp", type=str, nargs='+', default=None)
    parser.add_argument("--localized", type=int, default=0)


    # Load arguments from a YAML file if specified
    if config_path:
        arg_list = load_yaml_config(config_path)

    return parser.parse_args(arg_list)

