The purpose of this document is to provide details on how to use the python scripts included in the content folder. 

Limitations: 

Current version of create_identity_for_performance.py script requires the engine to be running in the background in order to validate the EULA and MetaHuman service authentication. In addition, an identity can only be created using footage capture data (using mesh as input parameter to create an identity is not supported)

The engine has to also be running in the background in order to export level sequence using process_performance.py. The rest of the functionality could be invoked in an unattended version of the engine.

Running the scripts individually:

create_capture_data.py

In order to execute create_capture_data.py the following command line parameters should be specified: --footage-path, --using-livelinkface-data and --storage-path. 
--footage path specifies the folder, where the footage for import is located on local machine
If the takes are of the type LiveLinkFace archive i.e. imported from ios device the --using-livelinkface-data parameter has to be specified
--storage-path is another optional parameter to specify where (within the current project) the imported assets should be stored. If this parameter is not specified, the assets will be stored in '/Game' directory

Example of running create_capture_data.py:

create_capture_data.py --footage-path D:/Data/iosTakes/ --using-livelinkface-data --storage-path /Game/MHA-Data/
create_capture_data.py --footage-path D:/Data/HMCTakes/


create_identity_for_performance.py

There are 2 ways an identity can be created - if a .dna file is present, the script has a create_identity_from_dna_file that could be invoked either by modifying existing script, or importing it to another script.

Alternatively this script could be invoked to create an identity from a specified frame that represents a neutral pose by providing the following parameters:
--neutral-frame a frame number that corresponds to neutral frame
--capture-data-path a path to capture data asset that points to the footage for neutral frame
--storage-path - as before, optional path to where identity would be stored in the project (/Game/MHA by default)
--body-index - another optional setting for MetaHuman body index, selected in a range 1-6 (1 by default)

Example of running create_identity_for_performance.py

create_identity_for_performance.py --neutral-frame 32 --capture-data-path /Game/MHA-Data/Source_Ingested/ku_iPhone-CD --storage-path /Game/MHA-Data

NOTE: when running this script, the identity name is hard-coded to test_identity_asset_name = 'KU_Test_Identity' inside the script. This should either be modified to a desired name or another parameter added as input


process_performance.py

In order to execute the process_performance.py script, the user has to specify the following parameters:
--identity-path a path to an identity asset
--capture-data-path a path to the capture data asset

There are a few optional parameters that could be also specified to customize the processing of performance and the output:

--storage-path as before, the location where performance asset will be stored
--start-frame specify the start frame for processing range (start of footage by default)
--end-frame specify the last frame for processing range(end of footage by default)

Example of running process_performance.py

process_performance.py --capture-data-path /Game/MHA-Data/Source_Ingested/ku_iPhone-CD --storage-path /Game/MHA-Data --identity-path /Game/MHA-Data/KU_Test_Identity

process_performance.py --capture-data-path /Game/MHA-Data/Source_Ingested/ku_iPhone-CD --identity-path /Game/MHA-Data/KU_Test_Identity


process_audio_performance.py

In order to execute the process_audio_performance.py script, the user has to specify the following parameter:
--soundwave-path a path to a sound wave asset

This optional parameter can be specified:
--storage-path as before, the location where performance asset will be stored

Example of running process_audio_performance.py

process_performance.py --soundwave-path /Game/MHA-Data/Example_SoundWave --storage-path /Game/MHA-Data


export_performance.py

In order to execute the export_performance.py script, the user has the specify:
--performance-path a path to a processed performance asset
--storage-path as before, the location where the anim sequence and level sequence assets will be stored

These optional parameters then specify which of the anim sequence or level sequences will be exported:

--export-anim-sequence specifying this will export the anim sequence
--export-identity-level-sequence this will export a level sequence with the identity in it
--export-MetaHuman-level-sequence this will export a level sequence with the target MetaHuman in it
--export-MetaHuman-side-view-level-sequence this will export a level sequence with the target MetaHuman rotated to get a side view

# The following parameter configures the target MetaHuman when exporting a level sequence with a MetaHuman in

--target-MetaHuman-path the path to the target MetaHuman blueprint asset if exporting a level sequence with a MetaHuman in it


Examples of running export_performance.py

export_performance.py --performance-path /Game/MHA-Data/ku_iPhone-CD_Performance --storage-path /Game/MHA-Data --export-anim-sequence --export-identity-level-sequence

export_performance.py --performance-path /Game/MHA-Data/ku_iPhone-CD_Performance --storage-path /Game/MHA-Data --export-MetaHuman-level-sequence --target-MetaHuman-path /Game/MetaHumans/Ada/BP_Ada


Rendering performances

The level sequence assets can be rendered out in the editor using movie render queue. Alternatively, you can use the command line using the UnrealEditor-Cmd.exe and the --LevelSequence parameter, for example:

"C:\path\to\your\installation\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" \path\to\your\project\project.uproject Map_To_Load -game -LevelSequence="/Game/path/to/the/LevelSequenceAsset" -windowed -Log -StdOut -allowStdOutLogVerbosity -Unattended


In addition these scripts could be used to run end-to-end processing as demonstrated:


# Copyright Epic Games, Inc. All Rights Reserved.

import unreal
import sys

content_directory = unreal.SystemLibrary.get_project_content_directory()
# Path to footage that is used to test the import script
path_to_footage = content_directory + '/AutoTestRawData/Footage/DeterministicTest'
# Path to exported dna and json file for identity creation
path_to_dna_data = content_directory + '/AutoTestRawData/DNA-Files/autotest-iphone-12-ku_1-CD/'
# Content path to metahuman blueprint asset
metahuman_bp_path = '/Game/MetaHumans/Ada/BP_Ada'


temp_storage = '/Game/TestScriptResults/'
test_range_start = 100
test_range_end = 180

def run_import_script():
    from create_capture_data import import_take_data_for_specified_device
    
    is_using_LLF_data = True
    LLF_archive_takes = import_take_data_for_specified_device(path_to_footage, is_using_LLF_data, temp_storage)

    return LLF_archive_takes

def run_create_identity_from_script(test_identity_asset_name):
    from create_identity_for_performance import create_identity_from_dna_file

    path_to_dna_file = path_to_dna_data + 'autotest-iphone-12-ku_1.dna'
    path_to_Json_file = path_to_dna_data + 'autotest-iphone-12-ku_1.json'
    prepare_for_performance = True
    create_identity_from_dna_file(path_to_dna_file, path_to_Json_file, temp_storage, test_identity_asset_name, prepare_for_performance)

    # Test if created identity is valid: 
    path_to_identity = temp_storage + test_identity_asset_name

    if unreal.EditorAssetLibrary.does_asset_exist(path_to_identity) == False:
        unreal.log_error('Failed to create identity: ' + test_identity_asset_name)

    identity: unreal.MetaHumanIdentity = unreal.load_asset(path_to_identity)
    face: unreal.MetaHumanIdentityFace = identity.get_or_create_part_of_class(unreal.MetaHumanIdentityFace)

    if face.has_dna_buffer() == False:
        unreal.log_error('Failed to apply a DNA to identity: ' + test_identity_asset_name)

    print('Successfully created MetaHuman identity')

def run_process_performance(identity_path, capture_data_path) ->unreal.MetaHumanPerformance :
    from process_performance import create_performance_asset
    from process_performance import process_shot

    performance_asset: unreal.MetaHumanPerformance = create_performance_asset(identity_path, capture_data_path, temp_storage)
    performance_asset.skip_diagnostics = True
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)

    process_shot(performance_asset, test_range_start, test_range_end)

    # Test processed performance:
    if performance_asset.contains_animation_data() == False:
        unreal.log_error('Failed to process performance: ' + performance_asset.get_name())

    if performance_asset.get_number_of_processed_frames() != test_range_end - test_range_start:
        unreal.log_error('Failed to process selected number of frames' + str(performance_asset.get_number_of_processed_frames()))

    print('Finished processing performance')
    return performance_asset
    
def export_performance(performance_asset: unreal.MetaHumanPerformance):
    from export_performance import run_anim_sequence_export
    from export_performance import run_identity_level_sequence_export
    from export_performance import run_meta_human_level_sequence_export

    anim_sequence = run_anim_sequence_export(performance_asset, temp_storage, "AS_ExampleSequence")
    print('Anim sequence exported')

    identity_level_sequence = run_identity_level_sequence_export(performance_asset, temp_storage, "LS_IdentityExampleSequence")
    print('Identity level sequence exported')

    meta_human_level_sequence = run_meta_human_level_sequence_export(performance_asset, metahuman_bp_path, temp_storage, "LS_MetaHumanExampleSequence")
    print('MetaHuman level sequence exported')    

def run():
    try:
        test_identity_asset_name = 'KU_Test_Identity'

        #Import the footage
        print('Running take import from script for testing')
        LLF_archive_source_paths = run_import_script()
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)

        #Create an identity using hard-coded path to DNA test files
        print('Running identity creation from script for testing')
        run_create_identity_from_script(test_identity_asset_name)

        #Create and process performance using pre-processed identity
        processed_identity = temp_storage + test_identity_asset_name
        capture_data_path = LLF_archive_source_paths[0]

        print('Running performance creation and processing from script')
        performance_asset = run_process_performance(processed_identity, capture_data_path)
        
        print('Exporting animation sequence and level sequence')
        export_performance(performance_asset)

        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
       
        print("Finished end-to-end workflow from test script")


    except Exception as e:
        print("ERROR: failed with error: %s" % e)
        sys.exit(1)

if __name__ == "__main__":
    run()