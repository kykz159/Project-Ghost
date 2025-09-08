# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import Paths

import CascadeToNiagaraHelperMethods as c2nUtils

class CascadeLocationBoneSocketConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLocationBoneSocket

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get all properties for the bone socket location module.
        # noinspection PyTypeChecker
        (source_type,
         universal_offset,
         source_locations,
         selection_method,
         b_update_positions_each_frame,
         b_orient_mesh_emitters,
         b_inherit_bone_velocity,
         inherit_velocity_scale,
         skel_mesh_actor_param_name,
         num_pre_selected_indices,
         editor_skel_mesh
         ) = ueFxUtils.get_particle_module_location_bone_socket_props(cascade_module)

        # choose whether to update each frame.
        if b_update_positions_each_frame:
            exec_category = ue.ScriptExecutionCategory.PARTICLE_UPDATE
        else:
            exec_category = ue.ScriptExecutionCategory.PARTICLE_SPAWN

        # add the skeletal mesh location module to the emitter.
        script_asset = ueFxUtils.create_asset_data(Paths.script_skeletal_mesh_location)
        script_args = ue.CreateScriptContextArgs(script_asset, [1, 1])
        skel_loc_script = emitter.find_or_add_module_script(
            "SkeletalMeshLocation",
            script_args,
            exec_category)

        # create a skeletal mesh DI to configure.
        skel_di = ue.NiagaraDataInterfaceSkeletalMesh()

        # set the sampling type.
        b_sample_bones = False
        b_sample_sockets = False
        if source_type == ue.LocationBoneSocketSource.BONESOCKETSOURCE_BONES:
            list_name = "FilteredBones"
            sample_mode_name = "Bone Sampling Mode"
            sample_mode_enum = Paths.enum_niagara_bone_sampling_mode
            b_sample_bones = True
            skel_loc_script.set_parameter(
                "Mesh Sampling Type",
                ueFxUtils.create_script_input_enum(
                    Paths.enum_niagara_skel_sampling_mode_full,
                    "Skeleton (Bones)"))
        elif source_type == ue.LocationBoneSocketSource.BONESOCKETSOURCE_SOCKETS:
            list_name = "FilteredSockets"
            sample_mode_name = "Socket Sampling Mode"
            sample_mode_enum = Paths.enum_niagara_socket_sampling_mode
            b_sample_sockets = True
            skel_loc_script.set_parameter(
                "Mesh Sampling Type",
                ueFxUtils.create_script_input_enum(
                    Paths.enum_niagara_skel_sampling_mode_full,
                    "Skeleton (Sockets)"))
        else:
            raise NameError("Encountered invalid location bone socket source "
                            "when converting cascade location bone socket "
                            "module!")

        # set the sampled position offset.
        skel_loc_script.set_parameter(
            "Sampled Position Offset",
            ueFxUtils.create_script_input_vector(universal_offset))
        
        sample_offset_sequential = False
        need_offsets = False
        location_offsets = []

        # copy the source locations to the skel mesh sampling DI.
        if len(source_locations) == 0:
            b_sampling_all = True
        else:
            b_sampling_all = False
            location_names = []
            b_logged_offset_warning = False
            for source_location in source_locations:
                location_names.append(source_location.get_editor_property("BoneSocketName"))
                location_offset = source_location.get_editor_property("Offset")
                location_offsets.append(location_offset)

            if any((offset != ue.Vector(0.0, 0.0, 0.0) for offset in location_offsets)):
                sample_offset_sequential = selection_method == ue.LocationBoneSocketSelectionMethod.BONESOCKETSEL_SEQUENTIAL
                need_offsets = True
                selection_method = ue.LocationBoneSocketSelectionMethod.BONESOCKETSEL_SEQUENTIAL

            skel_di.set_editor_property(list_name, location_names)

        # set the selection method.
        if selection_method == ue.LocationBoneSocketSelectionMethod.BONESOCKETSEL_SEQUENTIAL:
            sample_mode_prefix_str = "Direct "
        elif selection_method == ue.LocationBoneSocketSelectionMethod.BONESOCKETSEL_RANDOM:
            sample_mode_prefix_str = "Random "
        else:
            raise NameError("Encountered invalid location bone socket selection method when converting cascade "
                            "location bone socket module!")

        if b_sample_bones:
            if b_sampling_all:
                sample_mode_postfix_str = "(All Bones)"
            else:
                sample_mode_postfix_str = "(Filtered Bones)"
        elif b_sample_sockets:
            sample_mode_postfix_str = "(Filtered Sockets)"
        else:
            raise NameError("Encountered invalid selection when converting cascade location bone socket module; mode "
                            "was not sampling bones or sockets!")

        sample_mode_input = ueFxUtils.create_script_input_enum(
                sample_mode_enum,
                sample_mode_prefix_str + sample_mode_postfix_str)
        skel_loc_script.set_parameter(sample_mode_name, sample_mode_input)
            
        # choose whether to orient mesh emitters.
        # NOTE: this setting does nothing for this module.
        # if b_orient_mesh_emitters:

        # choose whether to inherit bone velocity.
        if b_inherit_bone_velocity:
            skel_loc_script.set_parameter(
                "Inherit Velocity (Scale)",
                ueFxUtils.create_script_input_float(inherit_velocity_scale),
                True,
                True)
        else:
            skel_loc_script.set_parameter(
                "Inherit Velocity (Scale)",
                ueFxUtils.create_script_input_float(1.0),
                True,
                False)

        # set the source skel mesh name
        # NOTE: niagara does not support skel mesh name lookup.
        # skel_mesh_actor_param_name

        # skip num_pre_selected_indices; Niagara does not support an equivalent
        # mode.

        # set the preview mesh.
        skel_di.set_editor_property("PreviewMesh", editor_skel_mesh)

        # set the di.
        skel_di_input = ueFxUtils.create_script_input_di(skel_di)
        skel_loc_script.set_parameter("Skeletal Mesh", skel_di_input)

        if need_offsets:
            if sample_offset_sequential:
                script_asset = ueFxUtils.create_asset_data(Paths.di_sequential_int)
                script_args = ue.CreateScriptContextArgs(script_asset)
                sequential_int_script = ueFxUtils.create_script_context(script_args)
                sequential_int_script.set_parameter("First Number", ueFxUtils.create_script_input_int(0))
                sequential_int_script.set_parameter("Last Number", ueFxUtils.create_script_input_int(len(location_offsets) - 1))
                sequential_int_input = ueFxUtils.create_script_input_dynamic(
                    sequential_int_script,
                    ue.NiagaraScriptInputType.INT)
                emitter.set_parameter_directly(
                    "Particles.SkeletonSamplingIndex",
                    sequential_int_input,
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
            else:
                script_asset = ueFxUtils.create_asset_data(Paths.di_random_range_int)
                script_args = ue.CreateScriptContextArgs(script_asset)
                random_range_int_script = ueFxUtils.create_script_context(script_args)
                random_range_int_script.set_parameter("Minimum", ueFxUtils.create_script_input_int(0))
                random_range_int_script.set_parameter("Maximum", ueFxUtils.create_script_input_int(len(location_offsets) - 1))
                random_range_input = ueFxUtils.create_script_input_dynamic(
                    random_range_int_script,
                    ue.NiagaraScriptInputType.INT)
                emitter.set_parameter_directly(
                    "Particles.SkeletonSamplingIndex",
                    random_range_input,
                    ue.ScriptExecutionCategory.PARTICLE_SPAWN)
            index_input1 = ueFxUtils.create_script_input_linked_parameter(
                "Particles.SkeletonSamplingIndex",
                ue.NiagaraScriptInputType.INT)
            index_input2 = ueFxUtils.create_script_input_linked_parameter(
                "Particles.SkeletonSamplingIndex",
                ue.NiagaraScriptInputType.INT)
            skel_loc_script.set_parameter('Bone / Socket Index', index_input1)
            script_asset = ueFxUtils.create_asset_data(Paths.di_select_vector_from_array)
            script_args = ue.CreateScriptContextArgs(script_asset)
            select_vector_script = ueFxUtils.create_script_context(script_args)
            array_sampling_mode = ueFxUtils.create_script_input_enum(Paths.enum_niagara_array_sampling_mode, "Direct Set")
            select_vector_script.set_parameter("Array Sampling Mode", array_sampling_mode)
            select_vector_script.set_parameter('Direct Array Index', index_input2)
            vector_array_di = ue.NiagaraDataInterfaceArrayFloat3()
            vector_array_di.float_data = location_offsets
            vector_array_di_input = ueFxUtils.create_script_input_di(vector_array_di)
            select_vector_script.set_parameter("Vector Selection Array", vector_array_di_input)
            select_vector_input = ueFxUtils.create_script_input_dynamic(
                select_vector_script,
                ue.NiagaraScriptInputType.VEC3)
            skel_loc_script.set_parameter('Sampled Position Offset', select_vector_input)

        elif selection_method == ue.LocationBoneSocketSelectionMethod.BONESOCKETSEL_SEQUENTIAL and not b_sampling_all and len(location_offsets) > 0:
            script_asset = ueFxUtils.create_asset_data(Paths.di_sequential_int)
            script_args = ue.CreateScriptContextArgs(script_asset)
            sequential_int_script = ueFxUtils.create_script_context(script_args)
            sequential_int_script.set_parameter("First Number", ueFxUtils.create_script_input_int(0))
            sequential_int_script.set_parameter("Last Number", ueFxUtils.create_script_input_int(len(location_offsets) - 1))
            sequential_int_input = ueFxUtils.create_script_input_dynamic(
                sequential_int_script,
                ue.NiagaraScriptInputType.INT)
            skel_loc_script.set_parameter('Bone / Socket Index', sequential_int_input)

        skel_loc_script.set_parameter('Orientation Sampling', ueFxUtils.create_script_input_enum(Paths.enum_niagara_attribute_sampling_apply_output, 'Output'))
