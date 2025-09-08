# Copyright Epic Games, Inc. All Rights Reserved.

import unreal
import argparse
import sys


def run_anim_sequence_export(performance_asset : unreal.MetaHumanPerformance, export_sequence_location : str, export_sequence_name : str):
    """
    Creates an Anim Sequence asset from the processed range of the input performance.
    Returns the newly created asset.

    Args
        performance_asset: performance with a processed range of frames to generate the sequence from
        export_sequence_location: content path for the new sequence asset
        export_sequence_name: asset name for the new sequence asset
    """
    performance_asset_name = performance_asset.get_name()
    unreal.log("Exporting animation sequence for Performance '{0}'".format(performance_asset_name))

    export_settings = unreal.MetaHumanPerformanceExportAnimationSettings()
    # This hides the dialog where the user can select the path to write the anim sequence
    export_settings.show_export_dialog = False

    # if the path and name are not set, the export will use the performance as a base name
    export_settings.package_path = export_sequence_location
    export_settings.asset_name = export_sequence_name

    # set face archetype as target skeleton
    target_skeleton = unreal.load_asset('/Game/MetaHumans/Common/Face/Face_Archetype_Skeleton.Face_Archetype_Skeleton')
    export_settings.target_skeleton_or_skeletal_mesh = target_skeleton

    # Enable to export the head rotation as curve data
    export_settings.enable_head_movement = True

    # Use export_range to select the whole sequence or only the processing range
    export_settings.export_range = unreal.PerformanceExportRange.PROCESSING_RANGE

    # Export the animation sequence from the performance using the given settings
    anim_sequence: unreal.AnimSequence = unreal.MetaHumanPerformanceExportUtils.export_animation_sequence(performance_asset, export_settings)
    
    if anim_sequence:
        unreal.log("Exported Anim Sequence {0}".format(anim_sequence.get_name()))
    else:
        unreal.log("Failed to export Anim Sequence")

    return anim_sequence


def run_identity_level_sequence_export(performance_asset : unreal.MetaHumanPerformance, export_sequence_location : str, export_sequence_name : str):
    """
    Creates a level Sequence asset from the processed range of the input performance.
    The level sequence contains the identity and cine camera.
    Returns the newly created asset.
    
    Args
        performance_asset: performance with a processed range of frames to generate the sequence from
        export_sequence_location: content path for the new sequence asset
        export_sequence_name: asset name for the new sequence asset
    """
    performance_asset_name = performance_asset.get_name()

    unreal.log("Exporting level sequence for performance '{0}'".format(performance_asset_name))
    level_sequence_export_settings = unreal.MetaHumanPerformanceExportLevelSequenceSettings()
    level_sequence_export_settings.show_export_dialog = False
    
    # if the path and name are not set, the export will use the performance as a base name
    level_sequence_export_settings.package_path = export_sequence_location
    level_sequence_export_settings.asset_name = export_sequence_name

    # customize various export settings
    level_sequence_export_settings.export_video_track = True
    level_sequence_export_settings.export_depth_track = False
    level_sequence_export_settings.export_audio_track = True
    level_sequence_export_settings.export_image_plane = False
    level_sequence_export_settings.export_identity = True
    level_sequence_export_settings.export_camera = True
    level_sequence_export_settings.apply_lens_distortion = True    
    level_sequence_export_settings.export_depth_mesh = False
    level_sequence_export_settings.export_control_rig_track = True
    level_sequence_export_settings.export_transform_track = False
    level_sequence_export_settings.keep_frame_range = True
    level_sequence_export_settings.target_meta_human_class = None
    level_sequence_export_settings.enable_meta_human_head_movement = True
    level_sequence_export_settings.export_range = unreal.PerformanceExportRange.WHOLE_SEQUENCE
    level_sequence_export_settings.curve_interpolation = unreal.RichCurveInterpMode.RCIM_LINEAR

    # export the level sequence
    exported_level_sequence: unreal.LevelSequence = unreal.MetaHumanPerformanceExportUtils.export_level_sequence(performance=performance_asset, export_settings=level_sequence_export_settings)

    if exported_level_sequence:
        unreal.log("Exported Level Sequence {0}".format(exported_level_sequence.get_name()))
    else:
        unreal.log("Failed to export Level Sequence")

    return exported_level_sequence


def run_meta_human_level_sequence_export(performance_asset : unreal.MetaHumanPerformance, target_meta_human_path : str, export_sequence_location : str, export_sequence_name : str):
    """
    Creates a level Sequence asset from the processed range of the input performance.
    The level sequence contains the metahuman and cine camera.
    Returns the newly created asset.
    
    Args
        performance_asset: performance with a processed range of frames to generate the sequence from
        target_meta_human_path: content path to the MetaHuman BP asset to target during the level sequence export
        export_sequence_location: content path for the new sequence asset
        export_sequence_name: asset name for the new sequence asset
    """
    performance_asset_name = performance_asset.get_name()

    unreal.log("Exporting level sequence for performance '{0}'".format(performance_asset_name))
    level_sequence_export_settings = unreal.MetaHumanPerformanceExportLevelSequenceSettings()
    level_sequence_export_settings.show_export_dialog = False
    
    # if the path and name are not set, the export will use the performance as a base name
    level_sequence_export_settings.package_path = export_sequence_location
    level_sequence_export_settings.asset_name = export_sequence_name

    # load the target metahuman blueprint asset
    if not target_meta_human_path:
        unreal.log_error("Unable to export level sequence with MetaHuman in. No target_MetaHuman_path set.")
        return None
    
    target_meta_human_bp_asset: unreal.Blueprint = unreal.load_asset(target_meta_human_path)
    target_meta_human_bp_generated_class = target_meta_human_bp_asset.generated_class()

    print(target_meta_human_bp_generated_class.get_name())
    
    # customize various export settings
    level_sequence_export_settings.export_video_track = True
    level_sequence_export_settings.export_depth_track = False
    level_sequence_export_settings.export_audio_track = True
    level_sequence_export_settings.export_image_plane = False
    level_sequence_export_settings.export_identity = False
    level_sequence_export_settings.export_camera = True
    level_sequence_export_settings.apply_lens_distortion = True    
    level_sequence_export_settings.export_depth_mesh = False
    level_sequence_export_settings.export_control_rig_track = True
    level_sequence_export_settings.export_transform_track = False
    level_sequence_export_settings.keep_frame_range = True
    level_sequence_export_settings.target_meta_human_class = target_meta_human_bp_asset
    level_sequence_export_settings.enable_meta_human_head_movement = True
    level_sequence_export_settings.export_range = unreal.PerformanceExportRange.WHOLE_SEQUENCE
    level_sequence_export_settings.curve_interpolation = unreal.RichCurveInterpMode.RCIM_LINEAR

    # export the level sequence
    exported_level_sequence: unreal.LevelSequence = unreal.MetaHumanPerformanceExportUtils.export_level_sequence(performance=performance_asset, export_settings=level_sequence_export_settings)
    if exported_level_sequence:
        unreal.log("Exported Level Sequence {0}".format(exported_level_sequence.get_name()))
    else:
        unreal.log("Failed to export Level Sequence")
    
    return exported_level_sequence


def is_meta_human_binding(binding : unreal.MovieSceneBindingProxy):
    # check for "Face" child possessable to determine if this is the metahuman binding
    for child in binding.get_child_possessables():
        if child.get_display_name() == 'Face':
            return True
    
    return False

def get_transform_section_of_meta_human_binding(level_sequence : unreal.LevelSequence):
    for object_binding in level_sequence.get_bindings():
        if is_meta_human_binding(object_binding):       
            transform_track = object_binding.find_tracks_by_exact_type(unreal.MovieScene3DTransformTrack)[0]
            return transform_track.get_sections()[0]
    
    return None


def run_meta_human_side_view_level_sequence_export(performance_asset : unreal.MetaHumanPerformance, target_meta_human_path : str, export_sequence_location : str, export_sequence_name : str):
    """
    Creates a level Sequence asset from the processed range of the input performance.
    The level sequence contains the metahuman rotated by 45 degrees to get a side view and cine camera.
    Returns the newly created asset.
    
    Args
        performance_asset: performance with a processed range of frames to generate the sequence from
        target_meta_human_path: content path to the MetaHuman BP asset to target during the level sequence export
        export_sequence_location: content path for the new sequence asset
        export_sequence_name: asset name for the new sequence asset
    """
        
    # export metahuman level sequence as above
    exported_level_sequence = run_meta_human_level_sequence_export(
        performance_asset=performance_asset, 
        target_meta_human_path=target_meta_human_path, 
        export_sequence_location=export_sequence_location, 
        export_sequence_name=export_sequence_name)
    
    if exported_level_sequence:
        # get MovieSceneSection for transform track of MetaHuman
        transform_section = get_transform_section_of_meta_human_binding(exported_level_sequence)

        # set Rotation.Z channel to 45 degrees to rotate the metahuman so we can see it from a side view
        for channel in transform_section.get_all_channels():
            if channel.get_editor_property("channel_name") == "Rotation.Z":
                channel.set_default(45)

    return exported_level_sequence
    


def run():
    """Main function to run for this module"""
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=
        'This script is used to export a MetaHuman Performance asset to an Anim Sequence and variations of a Level Sequence.'
        'The performance asset needs to be provided and has to have been processed.'
        )
    parser.add_argument("--performance-path", type=str, required=True, help="Content path to performance asset to be exported")
    parser.add_argument("--storage-path", type=str, default='/Game/', help="Content path where the assets should be stored, e.g. /Game/MHA-Data/")
    
    # Optional command line arguments specify which of the anim sequence or level sequences will be exported
    parser.add_argument("--export-anim-sequence", action="store_true", default=False, required=False, help="An optional parameter to enable anim sequence export")
    parser.add_argument("--export-identity-level-sequence", action="store_true", default=False, required=False, help="An optional parameter to enable level sequence export of the identity")
    parser.add_argument("--export-MetaHuman-level-sequence", action="store_true", default=False, required=False, help="An optional parameter to enable level sequence export of the metahuman")
    parser.add_argument("--export-MetaHuman-side-view-level-sequence", action="store_true", default=False, required=False, help="An optional parameter to enable level sequence export of the metahuman from a side view")

    # Parameter to configure the target MetaHuman if exporting a level sequence with a MetaHuman in
    parser.add_argument("--target-MetaHuman-path", type=str, required=False, help="An optional path to MetaHuman BP asset to target during the level sequence export")
    
    args = parser.parse_args()

    unreal.log("Loading performance asset {0}".format(args.performance_path))
    performance_asset = unreal.load_asset(args.performance_path)

    if args.export_anim_sequence:
        run_anim_sequence_export(
            performance_asset=performance_asset,
            export_sequence_location=args.storage_path,
            export_sequence_name=f'AS_{performance_asset.get_name()}')

    if args.export_identity_level_sequence:
        level_sequence = run_identity_level_sequence_export(
            performance_asset=performance_asset,
            export_sequence_location=args.storage_path,
            export_sequence_name=f'LS_Identity_{performance_asset.get_name()}')
        
        if level_sequence:
            unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_asset(level_sequence.get_outer().get_name(), False)
        
    if args.export_MetaHuman_level_sequence:
        level_sequence = run_meta_human_level_sequence_export(
            performance_asset=performance_asset,
            target_meta_human_path=args.target_MetaHuman_path,
            export_sequence_location=args.storage_path,
            export_sequence_name=f'LS_MetaHuman_{performance_asset.get_name()}')
        
        if level_sequence:
            unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_asset(level_sequence.get_outer().get_name(), False)
        
    if args.export_MetaHuman_side_view_level_sequence:
        level_sequence = run_meta_human_side_view_level_sequence_export(
            performance_asset=performance_asset,
            target_meta_human_path=args.target_MetaHuman_path,
            export_sequence_location=args.storage_path,
            export_sequence_name=f'LS_MetaHuman_side_view_{performance_asset.get_name()}')
        
        if level_sequence:
            unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_asset(level_sequence.get_outer().get_name(), False)


if __name__ == "__main__":
    run()
