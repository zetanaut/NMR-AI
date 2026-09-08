"""Exact acquisition grid and explicit input contracts for the 500-bin labs."""
import numpy as np

from baseline_data import load_acquisition

ACQUISITION = load_acquisition()
BINS = ACQUISITION["bins"]
FREQUENCY = ACQUISITION["start_mhz"] + np.arange(BINS)*ACQUISITION["step_mhz"]
FREQUENCY_HZ = FREQUENCY*1e6
SIMULATOR = "qmeter-complex-pake-500-v3"
MATCHED_SIMULATOR = "experiment-anchored-qmeter-pake-500-v1"
PREPROCESSING_VERSION = "nmr-inputs-500-v1"
CONTRACTS = {
    SIMULATOR: {"feature_mode": "te_area", "voltage_unit": "V", "group_column": "configuration_id"},
    MATCHED_SIMULATOR: {"feature_mode": "lineshape", "voltage_unit": "recorded units", "group_column": "source_scan_1based"},
}


def integration_weights(frequency_hz):
    f = np.asarray(frequency_hz, dtype=float)
    if f.ndim != 1 or len(f) < 2 or not np.isfinite(f).all() or np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("Need a finite, positive, increasing frequency grid")
    weights = np.empty_like(f)
    weights[0], weights[-1] = (f[1]-f[0])/2, (f[-1]-f[-2])/2
    weights[1:-1] = (f[2:]-f[:-2])/2
    return weights/f


def make_features(signals, calibration, baselines, *, mode="te_area"):
    """Subtract in float64, then build experimentally available input channels."""
    signals, baselines = np.asarray(signals, dtype=float), np.asarray(baselines, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != BINS or baselines.shape != signals.shape:
        raise ValueError("Need aligned (events,500) raw/reference arrays")
    if not len(signals) or not all(np.isfinite(a).all() for a in (signals, baselines)):
        raise ValueError("Raw/reference inputs must be finite and nonempty")
    residual = signals-baselines
    if mode == "te_area":
        calibration = np.asarray(calibration, dtype=float)
        if calibration.shape != (len(signals),) or not np.isfinite(calibration).all() or np.any(calibration == 0):
            raise ValueError("TE-area inputs require a finite nonzero calibration per event")
        residual = residual*calibration[:, None]*integration_weights(FREQUENCY_HZ)
    elif mode == "lineshape":
        if calibration is not None:
            raise ValueError("Lineshape inputs use raw/reference recorded units without TE calibration")
    else:
        raise ValueError("Unknown feature mode")
    features = np.stack([signals, residual], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Inputs overflow float32 features")
    return features


def input_contract(simulator):
    if simulator not in CONTRACTS:
        raise ValueError("Unsupported simulator version; generate a current 500-bin dataset")
    return {"version": PREPROCESSING_VERSION, "frequency_mhz": FREQUENCY.tolist(),
            **CONTRACTS[simulator]}


def validate_contract(contract):
    if not isinstance(contract, dict) or contract.get("version") != PREPROCESSING_VERSION:
        raise ValueError("Unsupported preprocessing version; train a current 500-bin model")
    if not np.array_equal(contract.get("frequency_mhz"), FREQUENCY):
        raise ValueError("Saved frequency grid does not match the confirmed acquisition")
    if not any(all(contract.get(k) == v for k, v in item.items()) for item in CONTRACTS.values()):
        raise ValueError("Saved feature mode, units or grouping do not match a supported contract")


def load_dataset(path, *, training=False, expected_contract=None, partition=False):
    """Validate the grid, units, labels and independent grouping before preprocessing.

    At inference a measured NPZ can omit simulator/labels/groups when it states
    feature_mode and voltage_unit explicitly. The checkpoint supplies the contract.
    """
    with np.load(path, allow_pickle=False) as data:
        if "frequency_mhz" not in data or not np.array_equal(data["frequency_mhz"], FREQUENCY):
            raise ValueError("Dataset frequency grid does not match the confirmed 500-bin acquisition")
        simulator = str(data["simulator"].item()) if "simulator" in data else None
        if simulator is not None:
            contract = input_contract(simulator)
        elif expected_contract is not None and not training and not partition:
            contract = expected_contract
            for key in ("feature_mode", "voltage_unit"):
                if key not in data or str(data[key].item()) != contract[key]:
                    raise ValueError(f"Measured inputs must explicitly declare {key} matching the model")
        else:
            raise ValueError("Training/partition data require a supported simulator version")
        validate_contract(contract)
        if expected_contract is not None and contract != expected_contract:
            raise ValueError("Dataset preprocessing, units or grouping do not match the saved model")
        for key in ("feature_mode", "voltage_unit"):
            if key in data and str(data[key].item()) != contract[key]:
                raise ValueError(f"Dataset {key} disagrees with its input contract")
        if not all(key in data for key in ("signals", "baselines")):
            raise ValueError("Dataset requires raw signals and corresponding references")
        calibration = data["calibration"] if "calibration" in data else None
        features = make_features(data["signals"], calibration, data["baselines"], mode=contract["feature_mode"])
        labels = data["P"] if "P" in data else None
        if labels is not None and (labels.shape != (len(features),) or not np.isfinite(labels).all() or np.any(abs(labels) > 1)):
            raise ValueError("P labels must be aligned finite fractions in [-1,1]")
        if (training or partition) and labels is None:
            raise ValueError("Training/partition evaluation requires simulator-known P labels")
        identifiers = {}
        if training or partition:
            for key in {"configuration_id", contract["group_column"]}:
                if key not in data:
                    raise ValueError(f"Dataset requires {key} to prevent group leakage")
                values = data[key]
                if values.shape != (len(features),) or values.dtype.kind not in "iu" or np.any(values < 0):
                    raise ValueError(f"{key} must contain aligned nonnegative integer identifiers")
                identifiers[key] = values.copy()
            if contract["group_column"] == "source_scan_1based":
                sources, configs = identifiers["source_scan_1based"], identifiers["configuration_id"]
                if np.any(sources < 1) or any(len(np.unique(sources[configs == group])) != 1 for group in np.unique(configs)):
                    raise ValueError("Each configuration must descend from one valid source scan")
    return {"features": features, "P": labels, "identifiers": identifiers,
            "contract": contract, "simulator": simulator}
