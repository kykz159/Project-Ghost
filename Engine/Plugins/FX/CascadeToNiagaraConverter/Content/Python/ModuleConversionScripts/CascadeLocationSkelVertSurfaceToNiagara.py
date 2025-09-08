# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import Paths

import CascadeToNiagaraHelperMethods as c2nUtils

class CascadeLocationSkelVertSurfaceConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLocationSkelVertSurface

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        (source_type,
         universal_offset,
         b_update_position_each_frame,
         b_orient_mesh_emitters,
         b_inherit_bone_velocity,
         inherit_velocity_scale,
         skel_mesh_actor_param_name,
         editor_skel_mesh,
         valid_associated_bones,
         b_enforce_normal_check,
         normal_to_compare,
         normal_check_tolerance_degrees,
         normal_check_tolerance,
         valid_material_indices,
         b_inherit_vertex_color,
         b_inherit_uv,
         inherit_uv_channel
         ) = ueFxUtils.get_particle_module_location_skel_vert_surface_props(cascade_module)

        if b_update_position_each_frame:
            exec_category = ue.ScriptExecutionCategory.PARTICLE_UPDATE
        else:
            exec_category = ue.ScriptExecutionCategory.PARTICLE_SPAWN

        script_asset = ueFxUtils.create_asset_data(Paths.script_skeletal_mesh_location)
        script_args = ue.CreateScriptContextArgs(script_asset, [1, 1])
        skel_loc_script = emitter.find_or_add_module_script(
            "SkeletalMeshLocation",
            script_args,
            exec_category)

        if source_type == ue.LocationSkelVertSurfaceSource.VERTSURFACESOURCE_VERT:
            mesh_sampling_type = 'Surface (Vertices)'
        else: # source_type == ue.LocationSkelVertSurfaceSource.VERTSURFACESOURCE_SURFACE
            mesh_sampling_type = 'Surface (Triangles)'
        skel_loc_script.set_parameter(
            "Mesh Sampling Type",
            ueFxUtils.create_script_input_enum(
                Paths.enum_niagara_skel_sampling_mode_full,
                mesh_sampling_type))

        # Surface sampling requires CPU access
        emitter.set_sim_target(ue.NiagaraSimTarget.CPU_SIM)

        skel_loc_script.set_parameter(
            "Sampled Position Offset",
            ueFxUtils.create_script_input_vector(universal_offset))

        skel_di = ue.NiagaraDataInterfaceSkeletalMesh()
        skel_di.set_editor_property("PreviewMesh", editor_skel_mesh)
        skel_di_input = ueFxUtils.create_script_input_di(skel_di)
        skel_loc_script.set_parameter("Skeletal Mesh", skel_di_input)

        velocity_sampling = 'Apply' if b_inherit_bone_velocity else 'Output'
        skel_loc_script.set_parameter('Velocity Sampling', ueFxUtils.create_script_input_enum(Paths.enum_niagara_attribute_sampling_apply_output, velocity_sampling))
        if b_inherit_bone_velocity:
            skel_loc_script.set_parameter(
                "Inherit Velocity (Scale)",
                ueFxUtils.create_script_input_float(inherit_velocity_scale))

        orientation_sampling = 'Apply' if b_orient_mesh_emitters else 'Output'
        skel_loc_script.set_parameter('Orientation Sampling', ueFxUtils.create_script_input_enum(Paths.enum_niagara_attribute_sampling_apply_output, orientation_sampling))
