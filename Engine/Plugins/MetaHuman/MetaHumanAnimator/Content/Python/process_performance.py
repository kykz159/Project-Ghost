# Copyright Epic Games, Inc. All Rights Reserved.

import unreal
import argparse
import sys


def create_performance_asset(path_to_identity : str, path_to_capture_data : str, save_performance_location : str) -> unreal.MetaHumanPerformance:
    """
    Returns a newly craeted MetaHuman Performance asset based on the input identity and capture data. 
    
    Args
        path_to_identity: content path to an existing MH Identity asset that is going to be used by the performance
        path_to_capture_data: content path to the capture data used to do the processing in performance
        save_performance_location: content path to store the new performance
    """
    capture_data_asset = unreal.load_asset(path_to_capture_data)
    identity_asset = unreal.load_asset(path_to_identity)
    performance_asset_name = "{0}_Performance".format(capture_data_asset.get_name())

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    performance_asset = asset_tools.create_asset(asset_name=performance_asset_name, package_path=save_performance_location, 
                                                 asset_class=unreal.MetaHumanPerformance, factory=unreal.MetaHumanPerformanceFactoryNew())

    # Use this style as set_editor_property doesn't trigger the PostEditChangeProperty required to setup the Performance asset
    performance_asset.set_editor_property("identity", identity_asset)
    performance_asset.set_editor_property("footage_capture_data", capture_data_asset)

    return performance_asset


def process_shot(performance_asset : unreal.MetaHumanPerformance, start_frame : int = None, end_frame : int = None):
    """
    Process the input performance and optionally export the processed range of frames as an AnimSequence asset.

    Args
        performance_asset: the performance to process
        start_frame, end_frame: set start/end frame property to change the processing range, the default range is set to entire shot
                                when modifying the range make sure start and end frames are vaild and not overlaping
                                upper limit frame will not be processed, so for limit of 10, frames 1-9 will be processed 
    """
    if start_frame is not None:
        performance_asset.set_editor_property("start_frame_to_process", start_frame)

    if end_frame is not None:
        performance_asset.set_editor_property("end_frame_to_process", end_frame)

    #Setting process to blocking will make sure the action is executed on the main thread, blocking it until processing is finished
    process_blocking = True
    performance_asset.set_blocking_processing(process_blocking)

    unreal.log("Starting MH pipeline for '{0}'".format(performance_asset.get_name()))
    startPipelineError = performance_asset.start_pipeline()
    if startPipelineError is unreal.StartPipelineErrorType.NONE:
        unreal.log("Finished MH pipeline for '{0}'".format(performance_asset.get_name()))
    else:
        unreal.log("Unknown error starting MH pipeline for '{0}'".format(performance_asset.get_name()))


def run():
    """Main function to run for this module"""
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=
        'This scirpt is used to create a MetaHuman Performance asset and process a shot '
        'level sequence export is currently not supported in headless mode and in oredr for this '
        'script to work, valid capture data asset and an identity that has been prepared for '
        'Performance have to be provided')
    parser.add_argument("--identity-path", type=str, required=True, help="Content path to identity asset to be used by the performance one")
    parser.add_argument("--capture-data-path", type=str, required=True, help="Content path to capture data asset to be used by the performance one")
    parser.add_argument("--storage-path", type=str, default='/Game/', help="Content path where the assets should be stored, e.g. /Game/MHA-Data/")

    # Optional command line arguments for exporting the processed animation 
    parser.add_argument("--start-frame", type=int, required=False, help="Set starting frame for performance processing")
    parser.add_argument("--end-frame", type=int, required=False, help="Set ending frame up to which the performance will be processed. Note Processing range is N-1")

    args = parser.parse_args()

    # the following params need to extra handling for default values
    processing_start_frame = None
    processing_end_frame = None
    if args.start_frame is not None:
        processing_start_frame = args.start_frame
    if args.end_frame is not None:
        processing_end_frame = args.end_frame

    performance_asset = create_performance_asset(
        path_to_identity=args.identity_path,
        path_to_capture_data=args.capture_data_path,
        save_performance_location=args.storage_path)
    process_shot(
        performance_asset=performance_asset,
        start_frame=processing_start_frame,
        end_frame=processing_end_frame)


if __name__ == "__main__":
    run()
