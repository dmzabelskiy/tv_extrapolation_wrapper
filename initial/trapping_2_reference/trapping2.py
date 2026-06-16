from xtr_estimator.estimation import plot_extrapolation_estimate
from xtr_estimator.masking import make_inclusion_mask
from xtr_estimator.xtr_maps import save_to_folder
from xtr_estimator.processing import cut_resolution


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


def apply_config_trap(datadir) -> Settings:
    name = "trapping_2"
    folderloc = datadir + name + "/"
    # Initialize the Settings object
    # This automatically runs all validation and @computed_field logic
    config = Settings(
        general=GeneralSettings(
            name_machine="trapping2",
            high_resolution_limit=1.45,
            comparison_type="triggered",
        ),
        input_files=InputFileSettings(
            map_dark=folderloc + "10_CD364A_ground.mtz",
            map_triggered=folderloc + "6_CD364A_473nm_RT_5sec_5mWt.mtz",
            pdb_dark=folderloc + "overall_best_refmac0_refmac0-coot-13_refine_075.pdb",
        ),
        map_processing=MapProcessingSettings(
            calculate_diffmap_before_f000=False, diffmap_type="it_tv"
        ),
        masking=MaskingSettings.simple(),
        plot=PlotSettings(
            solvent_density=0.3,
        ),
    )
    config.masking.exclude_solvent = False

    return config




def main():
    datadir = "/Users/sbielfel/Nextcloud2/time_resolved/data/o1_structures/"
    config = apply_config_trap(datadir)
    unscaled_dark, unscaled_triggered = get_maps(
        config.input_files, high_resolution_limit=config.general.high_resolution_limit
    )

    # ensure subtractibility
    unscaled_triggered.cell = unscaled_dark.cell
    unscaled_triggered = cut_resolution(
        unscaled_triggered, high_resolution_limit=config.general.high_resolution_limit
    )

    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    _, _, prediction_tuple = plot_extrapolation_estimate(
        diffmap, map_dark, inclusion_mask, config, compact=False
    )

    parameters = {
        "folder": config.general.output_folder,
        "xtr_prefix": "extrapolated",
        "diffmap_prefix": "diff",
    }
    save_to_folder(
        diffmap,
        map_dark,
        parameters,
        input_file_config=config.input_files,
        save_dict={"best_guess": 1 / prediction_tuple[0]},
        # rfree_flags = "FREE"
    )


if __name__ == "__main__":
    main()
