
from xtr_estimator.configuration import load_homepath
from xtr_estimator.estimation import plot_extrapolation_estimate
from xtr_estimator.main import execute_as_main
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.xtr_maps import save_to_folder


from xtr_estimator.configuration import (
    InputFileSettings,
    GeneralSettings,
    ColumnConfig,
    IntColumnConfig,
    Settings,
    MapProcessingSettings,
    PlotSettings,
    MaskingSettings,
)
from xtr_estimator.processing import get_maps, prepare_maps

def apply_config_5us(datadir) -> dict:
    name = "5us"
    folderloc = datadir + name + "/"
    # Initialize the Settings object
    # This automatically runs all validation and @computed_field logic
    config = Settings(
        general=GeneralSettings(
            name_machine=name,
            high_resolution_limit=1.8,
            comparison_type="triggered",
        ),
        input_files=InputFileSettings(
            map_dark=folderloc + "ground.mtz",
            map_triggered=folderloc + "5us.mtz",
            pdb_dark=folderloc + "olpvr1_xfel_dark_refine_007.pdb",
            columns_dark_ints=IntColumnConfig(
                ints_column="I",
                int_uncertainty_column="SIGI",
            ),
            columns_triggered_ints=IntColumnConfig(
                ints_column="I",
                int_uncertainty_column="SIGI",
            ),
            columns_are_ints=True
        ),
        map_processing=MapProcessingSettings( diffmap_type="it_tv"),
        masking=MaskingSettings.simple(),
        plot=PlotSettings(
            solvent_density=0.3,
        ),
    )
    return config


def extrapolate(config):
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    _, _, prediction_tuple = plot_extrapolation_estimate(
        diffmap, map_dark, inclusion_mask, config, compact=False
    )
    parameters = {"folder": config.general.output_folder, "xtr_prefix": "extrapolated", "diffmap_prefix": "diff"}
    save_to_folder(
        diffmap,
        map_dark,
        parameters,
        input_file_config=config.input_files,
        save_dict={"best_guess": 1 / prediction_tuple[0]},
        # rfree_flags = "FREE"
    )


def main():
    datadir = "/Users/sbielfel/Nextcloud2/time_resolved/data/o1_structures/"
    config = apply_config_5us(datadir)
    extrapolate(config) 

if __name__ == "__main__":
    main()  