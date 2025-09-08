# Copyright Epic Games, Inc. All Rights Reserved.

import unreal
import argparse
import sys


def convert_mood(mood: str):
    match mood:
        case 'Auto':
            return unreal.AudioDrivenAnimationMood.AUTO_DETECT
        case 'Neutral':
            return unreal.AudioDrivenAnimationMood.NEUTRAL
        case 'Happy':
            return unreal.AudioDrivenAnimationMood.HAPPY
        case 'Sad':
            return unreal.AudioDrivenAnimationMood.SAD
        case 'Disgust':
            return unreal.AudioDrivenAnimationMood.DISGUST
        case 'Anger':
            return unreal.AudioDrivenAnimationMood.ANGER
        case 'Surprise':
            return unreal.AudioDrivenAnimationMood.SURPRISE
        case 'Fear':
            return unreal.AudioDrivenAnimationMood.FEAR
        case _:
            raise RuntimeError(f'Unknown mood: {mood}')


def convert_output_controls(process_mask: str):
    match process_mask:
        case 'FullFace':
            return unreal.AudioDrivenAnimationOutputControls.FULL_FACE
        case 'MouthOnly':
            return unreal.AudioDrivenAnimationOutputControls.MOUTH_ONLY
        case _:
            raise RuntimeError(f'Unknown process mask: {process_mask}')


def convert_head_movement_mode(head_control_mode: str):
    match head_control_mode:
        case 'ControlRig':
            return unreal.PerformanceHeadMovementMode.CONTROL_RIG
        case 'TransformTrack':
            return unreal.PerformanceHeadMovementMode.TRANSFORM_TRACK
        case 'Disabled':
            return unreal.PerformanceHeadMovementMode.DISABLED
        case _:
            raise RuntimeError(f'Unknown head control mode: {head_control_mode}')


def create_performance_asset(
    path_to_sound_wave: str,
    save_performance_location: str,
    *,
    asset_name: str = None,
    mood: str = 'Auto',
    mood_intensity: float = 1.0,
    process_mask: str = 'FullFace',
    head_movement_mode: str = 'ControlRig'

) -> unreal.MetaHumanPerformance:
    """
    Returns a newly created MetaHuman Performance asset based on the input soundwave. 
    
    Args
        path_to_sound_wave: content path to a USoundWave asset that is going to be used by the performance
        save_performance_location: content path to store the new performance
    """
    sound_wave_asset = unreal.load_asset(path_to_sound_wave)
    performance_asset_name = "{0}_Performance".format(sound_wave_asset.get_name()) if not asset_name else asset_name

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    performance_asset = asset_tools.create_asset(asset_name=performance_asset_name, package_path=save_performance_location,
                                                 asset_class=unreal.MetaHumanPerformance, factory=unreal.MetaHumanPerformanceFactoryNew())

    # Use this style as set_editor_property doesn't trigger the PostEditChangeProperty required to setup the Performance asset
    performance_asset.set_editor_property("input_type", unreal.DataInputType.AUDIO)
    performance_asset.set_editor_property("audio", sound_wave_asset)

    solve_overrides = unreal.AudioDrivenAnimationSolveOverrides()
    solve_overrides.mood = convert_mood(mood)
    solve_overrides.mood_intensity = mood_intensity
    performance_asset.set_editor_property("audio_driven_animation_solve_overrides", solve_overrides)

    output_controls = convert_output_controls(process_mask)
    performance_asset.set_editor_property("audio_driven_animation_output_controls", output_controls)
    performance_asset.set_editor_property("head_movement_mode", convert_head_movement_mode(head_movement_mode))

    return performance_asset


def mood_intensity_type(value) -> float:
    float_value = float(value)
    if float_value < 0.0 or float_value > 1.0:
        raise argparse.ArgumentError(f'{float_value} is an invalid mood intensity value (must be between zero and one)')
    return float_value


def run():
    """Main function to run for this module"""
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="This script is used to create a MetaHuman Performance asset and process a shot from a USoundWave asset "
    )
    parser.add_argument("--soundwave-path", type=str, required=True, help="Content path to USoundWave asset to be used by the performance")
    parser.add_argument("--storage-path", type=str, default="/Game/", help="Content path where the assets should be stored, e.g. /Game/MHA-Data/")
    parser.add_argument("--mood", type=str, default="Auto",
                        choices=["Auto", "Neutral", "Happy", "Sad", "Disgust", "Anger", "Surprise", "Fear"],
                        help="Mood override")
    parser.add_argument("--mood-intensity", type=mood_intensity_type, default=1.0,
                        help="Mood intensity (0.0 - 1.0), has no effect with a Neutral mood")
    parser.add_argument("--process-mask", type=str, default="FullFace",
                        choices=["FullFace", "MouthOnly"],
                        help="Process mask to use (selectively enables output controls for the animation)")
    parser.add_argument("--head-movement-mode", type=str, default="ControlRig",
                        choices=["ControlRig", "TransformTrack", "Disabled"],
                        help="Head movement mode")

    args = parser.parse_args()

    performance_asset = create_performance_asset(
        path_to_sound_wave=args.soundwave_path,
        save_performance_location=args.storage_path,
        mood=args.mood,
        mood_intensity=args.mood_intensity,
        process_mask=args.process_mask,
        head_movement_mode=args.head_movement_mode
    )

    from process_performance import process_shot
    process_shot(performance_asset=performance_asset)


if __name__ == "__main__":
    run()
