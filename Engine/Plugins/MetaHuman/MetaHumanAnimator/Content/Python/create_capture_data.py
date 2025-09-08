# Copyright Epic Games, Inc. All Rights Reserved.

import unreal
import argparse
import os
import sys

from utility_functions_MH import get_or_create_asset


def prepare_ingest_root_path(capture_source_name : str, storage_path : str) -> str:
    """Returns an absolute path pointing to the (default) ingested location of the input capture source at the Content storage_path"""
    content_directory = unreal.SystemLibrary.get_project_content_directory()
    # Can't run os.path.join with a path that starts with '/'
    ingest_folder = storage_path.replace('/Game/', '')
    prepared_ingest_folder = os.path.join(content_directory, ingest_folder, "{0}_Ingested".format(capture_source_name))

    return unreal.Paths.normalize_directory_name(prepared_ingest_folder)


def prepare_asset_path(capture_source_name : str, storage_path : str) -> str:
    """Returns a Content path to the (default) ingested location of the input capture source at storage_path"""
    # Can't run os.path.join with a path that starts with '/'
    ingest_folder = storage_path.replace('/Game/', '')
    prepared_asset_folder = os.path.join('/Game', ingest_folder, "{0}_Ingested".format(capture_source_name))

    return unreal.Paths.normalize_directory_name(prepared_asset_folder)


def device_class_from_string(device_model_string: str) -> unreal.FootageDeviceClass:
    """ Utility for determining the Footage Device Class from the input device model string."""
    if device_model_string.startswith('iPhone') or device_model_string.startswith('iPad'):
        # dict with the mapping between iphone model number and the corresponding type value
        # iPad models are always treated as 'OtheriOSDevice'.
        _iphone_str_to_device = {
            'iPhone12,1': unreal.FootageDeviceClass.I_PHONE11_OR_EARLIER, # iPhone 11
            'iPhone12,3': unreal.FootageDeviceClass.I_PHONE11_OR_EARLIER, # iPhone 11 Pro
            'iPhone12,5': unreal.FootageDeviceClass.I_PHONE11_OR_EARLIER, # iPhone 11 Pro Max
            'iPhone13,1': unreal.FootageDeviceClass.I_PHONE12,            # iPhone 12 Mini
            'iPhone13,2': unreal.FootageDeviceClass.I_PHONE12,            # iPhone 12
            'iPhone13,3': unreal.FootageDeviceClass.I_PHONE12,            # iPhone 12 Pro
            'iPhone13,4': unreal.FootageDeviceClass.I_PHONE12,            # iPhone 12 Pro Max
            'iPhone14,2': unreal.FootageDeviceClass.I_PHONE13,            # iPhone 13 Pro
            'iPhone14,3': unreal.FootageDeviceClass.I_PHONE13,            # iPhone 13 Pro Max
            'iPhone14,4': unreal.FootageDeviceClass.I_PHONE13,            # iPhone 13 Mini
            'iPhone14,5': unreal.FootageDeviceClass.I_PHONE13,            # iPhone 13
            'iPhone14,7': unreal.FootageDeviceClass.I_PHONE14_OR_LATER,   # iPhone 14
            'iPhone14,8': unreal.FootageDeviceClass.I_PHONE14_OR_LATER,   # iPhone 14 Plus
            'iPhone15,2': unreal.FootageDeviceClass.I_PHONE14_OR_LATER,   # iPhone 14 Pro
            'iPhone15,3': unreal.FootageDeviceClass.I_PHONE14_OR_LATER,   # iPhone 14 Pro Max
            'iPhone15,4': unreal.FootageDeviceClass.I_PHONE14_OR_LATER,   # iPhone 15
            'iPhone15,5': unreal.FootageDeviceClass.I_PHONE14_OR_LATER,   # iPhone 15 Plus
            'iPhone16,1': unreal.FootageDeviceClass.I_PHONE14_OR_LATER,   # iPhone 15 Pro
            'iPhone16,2': unreal.FootageDeviceClass.I_PHONE14_OR_LATER,   # iPhone 15 Pro Max
        }
        return _iphone_str_to_device.get(device_model_string, unreal.FootageDeviceClass.OTHERI_OS_DEVICE)
    elif device_model_string == 'StereoHMC':
        return unreal.FootageDeviceClass.STEREO_HMC

    return unreal.FootageDeviceClass.UNSPECIFIED


def prepare_takes_for_ingest(takes : unreal.Array[unreal.MetaHumanTakeInfo]) -> unreal.Array[int]:
    """Helper to read the take indices from the input take infos"""
    take_ids = unreal.Array(int)

    for take_info in takes:
        take_id = take_info.get_editor_property('id')
        take_ids.append(take_id)

    return take_ids


def create_capture_data_assets_for_imported_takes(
        capture_source: unreal.MetaHumanCaptureSource, imported_takes: list, storage_path: str) -> list:
    """
    Create the FootageCaptureData for each take in imported_takes of the input capture_source.
    Args
        capture_source: the capture source to read the takes from
        imported_takes: a list of takes to create the capture data for
        storage_path: a content path to create the new data under it
    """
    created_capture_data_assets = []
    for take in imported_takes:
        take_info: unreal.MetaHumanTakeInfo = capture_source.get_take_info(take.take_id)
        package_path = prepare_asset_path(unreal.Paths.make_platform_filename(capture_source.get_name()), storage_path)
        new_asset_name = take_info.name + '-CD'
        capture_data_asset = get_or_create_asset(new_asset_name, package_path, unreal.FootageCaptureData, unreal.FootageCaptureDataFactory())

        created_capture_data_assets.append(package_path + "/" + take_info.name + '-CD')
        # Setting all relevant parameters for Capture Data
        image_sequence_timecode_utils = unreal.ImageSequenceTimecodeUtils()

        capture_data_asset.image_sequences.clear()
        capture_data_asset.depth_sequences.clear()

        for view in take.views:
            if view.video_timecode_present:
                image_sequence_timecode_utils.set_timecode_info(view.video_timecode, view.video_timecode_rate, view.video)
            capture_data_asset.image_sequences.append(view.video)

            if view.depth_timecode_present:
                image_sequence_timecode_utils.set_timecode_info(view.depth_timecode, view.depth_timecode_rate, view.depth)
            capture_data_asset.depth_sequences.append(view.depth)
               
        capture_data_asset.audio_tracks = [take.audio, ]
        capture_data_asset.camera_calibrations = [take.camera_calibration, ]
        capture_data_asset.capture_excluded_frames = take.capture_excluded_frames      

        capture_data_asset_metadata = capture_data_asset.get_editor_property("metadata")
        capture_data_asset_metadata.device_class = device_class_from_string(take_info.device_model)
        capture_data_asset_metadata.device_model_name = take_info.device_model
        capture_data_asset_metadata.frame_rate = take_info.frame_rate

    return created_capture_data_assets


def import_take_data_for_specified_device(footage_path : str, using_LLF_data : bool, storage_path : str) -> list:
    """"
    Importing footage, generating relevant capture assets as part of import process and 
    Returns a list of created capture data assets.
    Args
        footage_path: absolute path to a folder on disk containing the footage
        using_LLF_data: whether the footage is from LiveLinkFace Archive or not
        storage_path: a project content path to create the capture data under it
    """
    # create a capture source to import the footage
    # note the Python API is calling a synchronous version of capture source that does not get stored as asset in content browser
    capture_source = unreal.MetaHumanCaptureSourceSync()
    capture_source_type = unreal.MetaHumanCaptureSourceType.LIVE_LINK_FACE_ARCHIVES if using_LLF_data is True else unreal.MetaHumanCaptureSourceType.HMC_ARCHIVES
    capture_source.set_editor_property('capture_source_type', capture_source_type)
    dir_path = unreal.DirectoryPath()
    dir_path.set_editor_property("path", footage_path)
    capture_source.set_editor_property('storage_path', dir_path)
    capture_data_asset_list = []

    if capture_source.can_startup():
        capture_source.startup()

        # populating the list of available takes for import at the specified location
        takes = capture_source.refresh()
        take_ids = prepare_takes_for_ingest(takes)

        # setup the paths to the ingested files
        capture_source_asset_name = unreal.Paths.make_platform_filename(capture_source.get_name())
        capture_source_target_ingest_directory = prepare_ingest_root_path(capture_source_asset_name, storage_path)
        capture_source_target_asset_directory = prepare_asset_path(capture_source_asset_name, storage_path)
        capture_source.set_target_path(capture_source_target_ingest_directory, capture_source_target_asset_directory)

        # running the import process for all takes in a specified location
        imported_takes = capture_source.get_takes(take_ids)
        capture_data_asset_list = create_capture_data_assets_for_imported_takes(capture_source, imported_takes, storage_path)

        capture_source.shutdown()

    else:
        unreal.log_error(f"Failed to import footage from {footage_path}")

    return capture_data_asset_list


def run():
    """Main function to run this script"""
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=
        'This script is used to import takes for specified device (either iPhone or HMC). '
        'A temporary capture source asset is created and all the takes, the source is pointed to, '
        'are imported. As part of the import process all relevant assets are created and '
        'a list of capture data source assets are returned. These capture source assets could '
        'be further used by identity creation or performance processing scripts.')
    parser.add_argument("--footage-path", type=str, required=True, help="An absolute path to a folder on disk, containing footage from the capture device")
    parser.add_argument("--using-livelinkface-data", action="store_true", default=False, help="Set if data comes from LiveLinkFace Archive, otherwise data will be treated as if it comes from HMC")
    parser.add_argument("--storage-path", type=str, default='/Game/', help="Project Content path where the assets should be stored, e.g. /Game/MHA-Data/")
    args = parser.parse_args()

    import_take_data_for_specified_device(
        footage_path=args.footage_path,
        using_LLF_data=args.using_livelinkface_data,
        storage_path=args.storage_path)


if __name__ == "__main__":
    run()
