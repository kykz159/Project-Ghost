# Copyright Epic Games, Inc. All Rights Reserved.

import unreal
import argparse
import sys
import utility_functions_MH

global_identity_asset_name = ""
global_identity_storage_location = ""


def prepare_identity_for_performance(identity_asset : unreal.MetaHumanIdentity):
    """Runs the 'Prepare for Performance' step for the input identity asset"""
    face: unreal.MetaHumanIdentityFace = identity_asset.get_or_create_part_of_class(unreal.MetaHumanIdentityFace)
    face.run_predictive_solver_training()
    unreal.log("Created Identity could now be used to process performance")


def process_autorig_service_response(dna_applied_success : bool):
    """Callback to process the response from the auto rigging service (Mesh to MetaHuman)"""
    global global_identity_asset_name, global_identity_storage_location
    unreal.log("Cleaning up the delegate for '{0}'".format(global_identity_asset_name))

    identity_asset = unreal.load_asset(global_identity_storage_location + '/' + global_identity_asset_name)
    identity_asset.on_auto_rig_service_finished_dynamic_delegate.remove_callable(process_autorig_service_response)
    global_identity_asset_name = ''
    if(dna_applied_success):
        unreal.log("Preparing Identity {} for performance...".format(identity_asset.get_name()))
        prepare_identity_for_performance(identity_asset)
    else:
        unreal.log_error("Failed to retrieve the DNA from Autorig service")


def create_identity_from_frame(neutral_frame : int, capture_source_asset_path : str, asset_storage_location : str,
                               identity_asset_name : str, prepare_for_performance : bool, body_index : int):
    """
    Creates an identity for selected frame that represents a neutral pose. If prepare for performance variable is true,
    A back-end service is invoked to get the dna data for conformed mesh. After the back-end service returns data the 
    response is processed and identity is prepared for performance. 
    !!! Note that back-end service runs as a separate thread and identity can not be used in performnace before it's been processed

    Args
        neutral_frame: an index to frames of the input capture_source_asset_path, should correspond to a neutral expression
        capture_source_asset_path: path to a capture source 
        asset_storage_location:  path to where the identity asset should be created
        identity_asset_name: the name of the new identity asset
        prepare_for_performance: whether to run "Prepare for Performance" after creating the identity
        body_index: the body type index, should be selected as an int in a [1-6] range for corresponding presets
    """

    global global_identity_asset_name, global_identity_storage_location
    global_identity_asset_name = identity_asset_name
    global_identity_storage_location = asset_storage_location

    if not unreal.EditorAssetLibrary.does_asset_exist(capture_source_asset_path):
        unreal.log_error(f"Could not locate Capture Data Source at provided location: {capture_source_asset_path}")

    capture_data_asset = unreal.load_asset(capture_source_asset_path)
    mh_identity_asset: unreal.MetaHumanIdentity = utility_functions_MH.create_or_recreate_asset(identity_asset_name, asset_storage_location, 
                                                                                                        unreal.MetaHumanIdentity, unreal.MetaHumanIdentityFactoryNew())
    mh_identity_asset.get_or_create_part_of_class(unreal.MetaHumanIdentityFace)

    if not mh_identity_asset.is_logged_in_to_service():
        mh_identity_asset.log_in_to_auto_rig_service()

    # add a neutral pose to the identity face
    face: unreal.MetaHumanIdentityFace = mh_identity_asset.get_or_create_part_of_class(unreal.MetaHumanIdentityFace)
    pose: unreal.MetaHumanIdentityPose = unreal.new_object(type=unreal.MetaHumanIdentityPose, outer=face)

    face.add_pose_of_type(unreal.IdentityPoseType.NEUTRAL, pose)
    pose.set_capture_data(capture_data_asset)
    pose.fit_eyes = True # we fit the eyes for footage to MetaHuman

    # create a promoted frame for the neutral pose
    pose.load_default_tracker()
    frame, _ = pose.add_new_promoted_frame()
    if frame is None:
        unreal.log_error("Failed to add promoted frame")
    frame.is_front_view = True
    frame.set_navigation_locked(True)
    frame.frame_number = neutral_frame

    # run the tracking on the promoted neutral frame
    if unreal.PromotedFrameUtils.initialize_contour_data_for_footage_frame(pose, frame):
        camera_name = pose.get_editor_property("camera")
        image_path = unreal.PromotedFrameUtils.get_image_path_for_frame(capture_data_asset, camera_name, neutral_frame, True, pose.timecode_alignment)
        depth_path = unreal.PromotedFrameUtils.get_image_path_for_frame(capture_data_asset, camera_name, neutral_frame, False, pose.timecode_alignment)

        # retreiving image from disk and storing it in an array
        image_size, local_samples = unreal.PromotedFrameUtils.get_promoted_frame_as_pixel_array_from_disk(image_path)

        if(image_size.x > 0 and image_size.y > 0) :
            # Make sure the pipeline is running synchronously with no progress indicators
            show_progress = False
            mh_identity_asset.set_blocking_processing(True)
            # Running tracking pipeline to get contour data for image retrieved from disk       
            mh_identity_asset.start_frame_tracking_pipeline(local_samples, image_size.x, image_size.y, depth_path, pose, frame, show_progress)

            log_only_no_dialogue = True
            # Running face conformation for template mesh
            if unreal.MetaHumanIdentity.handle_error(face.conform(), log_only_no_dialogue) :
                unreal.log("Face has been conformed")
                body: unreal.MetaHumanIdentityBody = mh_identity_asset.get_or_create_part_of_class(unreal.MetaHumanIdentityBody)
                body.body_type_index = body_index

                if prepare_for_performance:
                    if mh_identity_asset.is_logged_in_to_service():
                        unreal.log("Calling AutoRig service to create a DNA for identity")
                        mh_identity_asset.on_auto_rig_service_finished_dynamic_delegate.add_callable(process_autorig_service_response)
                        mh_identity_asset.create_dna_for_identity(log_only_no_dialogue)
                    else:
                        unreal.log_error("Please make sure you are logged in to MetaHuman service")
            else:
                unreal.log_error("Failed to conform the face")
    else:
        unreal.log_error("Failed to initialize contour data. Please make sure valid frame is selected")


def create_identity_from_dna_file(path_to_dna_file, path_to_Json, asset_storage_location, identity_name, prepare_for_performance: bool) :
    """
    Creates a new MH Identity asset based on the input DNA file.

    Args
        path_to_dna_file: fila path to the DNA file on disk
        path_to_Json: fila path to the corresponding json file for the DNA one
        asset_storage_location: content path for this project to create the new Identity asset under
        identity_name: the name to use for the new Identity asset
        prepare_for_performance: whether to run "Prepare for performance" after identity creation
    """
    mh_identity_asset = utility_functions_MH.create_or_recreate_asset(identity_name, asset_storage_location, 
                                                                      unreal.MetaHumanIdentity, unreal.MetaHumanIdentityFactoryNew())

    # create the face part of the identity
    mh_identity_asset.get_or_create_part_of_class(unreal.MetaHumanIdentityFace)

    # import the DNA file and apply to the identity asset
    dna_type = unreal.DNADataLayer.ALL
    import_error: unreal.IdentityErrorCode = mh_identity_asset.import_dna_file(path_to_dna_file, dna_type, path_to_Json)
    if import_error == unreal.IdentityErrorCode.NONE:
        if prepare_for_performance is True:
            prepare_identity_for_performance(mh_identity_asset)
    else:
        unreal.log_error('Selected DNA and Json files are not compatible with this plugin')


def run():
    """Main function to run for this module"""
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=
        'This script loads a capture data source of type footage, creates an identity with a passed command line argument frame number '
        'as promoted frame. Initializes contour data from the config and runs the tracking pipeline for that frame. '
        'The data is then used to conform the template mesh. The back-end AutoRig service is invoked to retrieve a DNA, which is applied '
        'to the skeletal mesh. At which point the identity is prepared for performance. '
        'The user must be connected to AutoRig service prior to running this script. '
        'In addition, a frame number for a neutral pose must be supplied as an argument to the script.')
    parser.add_argument("--asset-name", type=str, required=True, help="Name for the new identity asset")
    parser.add_argument("--storage-path", type=str, default='/Game/MHA', help="Content path where the assets should be stored, e.g. /Game/MHA-Data/")
    parser.add_argument("--capture-data-path", type=str, required=True, help="Content path to a capture data asset used for creating the neutral pose")
    parser.add_argument("--neutral-frame", type=int, required=True, help="Frame number that corresponds to neutral pose")
    parser.add_argument("--body-index", type=int, default=1, choices=range(1, 7), help="Body index of the created MetaHuman Identity, from the list of available MH body types")
    # parser.add_argument("--no-mesh2MH", action="store_true", default=False, help="Do not run Mesh to MetaHuman")

    args = parser.parse_args()

    create_identity_from_frame(
        neutral_frame=args.neutral_frame, 
        capture_source_asset_path=args.capture_data_path,
        asset_storage_location=args.storage_path,
        identity_asset_name=args.asset_name,
        prepare_for_performance=True,
        body_index=args.body_index)


if __name__ == "__main__":
    run()
