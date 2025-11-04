"""Count model parameters from YAML configuration files.

This script is designed to be used by LLMs for iterative model architecture design.
It takes a model configuration YAML file and outputs the total parameter count.

USAGE:
    python scripts/count_model_params.py <path_to_config.yaml>

EXAMPLE:
    python scripts/count_model_params.py configs/models/ealstm_baseline.yaml

OUTPUT EXAMPLE:
    Model: ealstm
    Config: configs/models/ealstm_baseline.yaml
    Total parameters: 275,716
    Trainable parameters: 275,716
    Non-trainable parameters: 0

IMPORTANT NOTES FOR LLMs:

1. MODEL HYPERPARAMETERS:
   - All model-specific hyperparameters MUST be in the `model.overrides` section
   - Parameters NOT specified in overrides will use default values from the model's config class
   - This is INTENTIONAL - you don't need to specify every parameter
   - The script will warn you about which parameters are using defaults

2. REQUIRED CONFIG SECTIONS:
   Your YAML config must include:

   features:
     forcing: [list of forcing features]  # Time-varying features
     static: [list of static features]    # Time-invariant features
     target: streamflow                    # Target variable name
     future: [list of future features]     # Optional - known future covariates

   sequence:
     input_length: 365                     # Lookback window length
     output_length: 1                      # Forecast horizon length

   data_preparation:
     mode: forecast                        # "forecast" or "simulation"
     is_autoregressive: true               # Include target in inputs?
     include_dates: true                   # Include timestamps?

   model:
     type: ealstm                          # Model architecture name
     overrides:                            # Model hyperparameters
       hidden_size: 256
       num_layers: 1
       # ... other hyperparameters

3. HOW DIMENSIONS ARE COMPUTED:
   - input_size = len(features.forcing)
     * This is ALWAYS just the count of forcing features
     * The target is included in forcing when is_autoregressive=true
     * The target is excluded from forcing when is_autoregressive=false

   - static_size = len(features.static)

   - future_input_size:
     * If features.future is specified: len(features.future)
     * If mode="simulation": 0 (no future features in simulation)
     * Otherwise: uses model's default (typically max(1, input_size-1))

   - input_len = sequence.input_length
   - output_len = sequence.output_length

4. SUPPORTED MODEL TYPES:
   - ealstm: Entity-Aware LSTM
   - tide: Time-series Dense Encoder
   - tsmixer: Time Series Mixer
   - tft: Temporal Fusion Transformer
   - mamba: Mamba architecture
   - naive_last_value: Naive baseline

5. COMMON HYPERPARAMETERS BY MODEL:

   EALSTM:
     hidden_size: 256
     num_layers: 1
     bidirectional: false
     dropout: 0.2

   TiDE:
     hidden_size: 128
     num_encoder_layers: 2
     num_decoder_layers: 2
     decoder_output_size: 16
     temporal_decoder_hidden_size: 32
     dropout: 0.1

   TSMixer:
     n_blocks: 2
     hidden_size: 64
     ff_size: 256
     dropout: 0.1

   TFT:
     hidden_size: 16
     lstm_layers: 2
     attention_heads: 4
     dropout: 0.1

   Mamba:
     d_model: 128
     n_layers: 4
     d_state: 16
     d_conv: 4
     expand: 2
     decoder_hidden_size: 128

6. WARNINGS VS ERRORS:
   - WARNINGS: Parameters using default values (this is OK!)
   - ERRORS: Missing required config sections, invalid model type, etc.

7. EXIT CODES:
   - 0: Success
   - 1: Error (invalid config, missing sections, model creation failed)

TYPICAL WORKFLOW FOR LLMs:
1. Create a config YAML with desired features and sequences
2. Add model.type and initial model.overrides
3. Run this script to check parameter count
4. Adjust hyperparameters in model.overrides
5. Repeat steps 3-4 until target parameter count is achieved
"""

import sys
from pathlib import Path
from typing import Any

import yaml
from transfer_learning_publication.models import ModelFactory


def count_parameters(model) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch Lightning model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load and validate YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required sections are missing
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["features", "sequence", "data_preparation", "model"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    # Validate features subsections
    required_features = ["forcing", "static", "target"]
    missing_features = [f for f in required_features if f not in config["features"]]
    if missing_features:
        raise ValueError(f"Missing required features subsections: {missing_features}")

    # Validate sequence fields
    required_sequence = ["input_length", "output_length"]
    missing_sequence = [f for f in required_sequence if f not in config["sequence"]]
    if missing_sequence:
        raise ValueError(f"Missing required sequence fields: {missing_sequence}")

    # Validate data_preparation fields
    required_data_prep = ["mode", "is_autoregressive"]
    missing_data_prep = [f for f in required_data_prep if f not in config["data_preparation"]]
    if missing_data_prep:
        raise ValueError(f"Missing required data_preparation fields: {missing_data_prep}")

    # Validate model section
    if "type" not in config["model"]:
        raise ValueError("Missing required field: model.type")

    return config


def compute_dimensions(config: dict[str, Any]) -> dict[str, int]:
    """Compute model input dimensions from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with dimension values
    """
    features = config["features"]
    sequence = config["sequence"]
    data_prep = config["data_preparation"]

    # Input size is always the count of forcing features
    input_size = len(features["forcing"])

    # Static size
    static_size = len(features["static"])

    # Future input size
    future_features = features.get("future", [])
    if future_features:
        # Explicit future features specified
        future_input_size = len(future_features)
    elif data_prep["mode"] == "simulation":
        # Simulation mode has no future features
        future_input_size = 0
    else:
        # Let model use its default (typically max(1, input_size-1))
        # We don't set it explicitly here
        future_input_size = None

    return {
        "input_size": input_size,
        "static_size": static_size,
        "future_input_size": future_input_size,
        "input_len": sequence["input_length"],
        "output_len": sequence["output_length"],
    }


def build_model_config(config: dict[str, Any], dimensions: dict[str, int]) -> dict[str, Any]:
    """Build model configuration dictionary for ModelFactory.

    Args:
        config: Full configuration dictionary
        dimensions: Computed dimensions

    Returns:
        Model configuration dict
    """
    # Start with computed dimensions
    model_config = {
        "input_size": dimensions["input_size"],
        "static_size": dimensions["static_size"],
        "input_len": dimensions["input_len"],
        "output_len": dimensions["output_len"],
    }

    # Add future_input_size only if we computed it
    if dimensions["future_input_size"] is not None:
        model_config["future_input_size"] = dimensions["future_input_size"]

    # Merge with model overrides if present
    overrides = config["model"].get("overrides", {})
    model_config.update(overrides)

    # Remove loss function configuration - it doesn't affect parameter count
    # and some losses (like basin_nse) require data that we don't have
    model_config.pop("loss_fn", None)
    model_config.pop("loss_fn_kwargs", None)

    # Remove scheduler configuration - doesn't affect parameter count either
    model_config.pop("scheduler", None)
    model_config.pop("scheduler_kwargs", None)

    return model_config


def check_default_usage(model_type: str, provided_config: dict[str, Any]) -> list[str]:
    """Check which standard hyperparameters are using default values.

    Args:
        model_type: Model architecture type
        provided_config: Configuration dict passed to model

    Returns:
        List of warning messages about parameters using defaults
    """
    warnings = []

    # Common parameters across all models
    common_params = ["learning_rate", "dropout"]

    # Model-specific important parameters
    model_specific = {
        "ealstm": ["hidden_size", "num_layers", "bidirectional"],
        "tide": ["hidden_size", "num_encoder_layers", "num_decoder_layers"],
        "tsmixer": ["n_blocks", "hidden_size", "ff_size"],
        "tft": ["hidden_size", "lstm_layers", "attention_heads"],
        "mamba": ["d_model", "n_layers", "d_state"],
    }

    # Parameters to check
    params_to_check = common_params + model_specific.get(model_type, [])

    # Skip dimensions - these are always computed
    skip_params = {"input_size", "static_size", "future_input_size", "input_len", "output_len"}

    using_defaults = []
    for param in params_to_check:
        if param not in skip_params and param not in provided_config:
            using_defaults.append(param)

    if using_defaults:
        warnings.append(f"WARNING: The following parameters are using default values: {', '.join(using_defaults)}")
        warnings.append("         This is fine if intentional. Add to model.overrides to set explicitly.")

    return warnings


def main() -> None:
    """Main entry point for parameter counting script."""
    # Check arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("ERROR: Expected one or two arguments")
        print()
        print("Usage:")
        print("    python scripts/count_model_params.py <path_to_config.yaml> [--debug]")
        print()
        print("Example:")
        print("    python scripts/count_model_params.py configs/models/ealstm_baseline.yaml")
        print("    python scripts/count_model_params.py configs/models/ealstm_baseline.yaml --debug")
        sys.exit(1)

    config_path = sys.argv[1]
    debug_mode = len(sys.argv) == 3 and sys.argv[2] == "--debug"

    try:
        # Load and validate config
        config = load_yaml_config(config_path)

        # Compute dimensions
        dimensions = compute_dimensions(config)

        # Build model config
        model_config = build_model_config(config, dimensions)

        # Get model type
        model_type = config["model"]["type"]

        # Check for default parameter usage
        warnings = check_default_usage(model_type, model_config)

        # Print debug info if requested
        if debug_mode:
            print("=" * 80)
            print("DEBUG: Model Configuration")
            print("=" * 80)
            print(f"Dimensions:")
            print(f"  input_size: {dimensions['input_size']}")
            print(f"  static_size: {dimensions['static_size']}")
            print(f"  future_input_size: {dimensions['future_input_size']}")
            print(f"  input_len: {dimensions['input_len']}")
            print(f"  output_len: {dimensions['output_len']}")
            print(f"\nModel Config Dict:")
            for key, value in sorted(model_config.items()):
                print(f"  {key}: {value}")
            print("=" * 80)
            print()

        # Create model
        factory = ModelFactory()
        model = factory.create_from_dict(model_type, model_config)

        # Count parameters
        counts = count_parameters(model)

        # Print results
        print(f"Model: {model_type}")
        print(f"Config: {config_path}")
        print(f"Total parameters: {counts['total']:,}")
        print(f"Trainable parameters: {counts['trainable']:,}")
        print(f"Non-trainable parameters: {counts['non_trainable']:,}")

        # Print warnings if any
        if warnings:
            print()
            for warning in warnings:
                print(warning)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
