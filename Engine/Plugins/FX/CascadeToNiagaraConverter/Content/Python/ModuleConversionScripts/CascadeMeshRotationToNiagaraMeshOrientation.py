# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeMeshRotationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleMeshRotation

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get all properties from the cascade mesh rotation module
        # noinspection PyTypeChecker
        (rotation,
         b_inherit_parent_rotation
         ) = ueFxUtils.get_particle_module_mesh_rotation_props(cascade_module)

        rotation_input = c2nUtils.create_script_input_for_distribution(rotation)
        emitter.set_parameter_directly(
            "Particles.MeshRotationEuler",
            rotation_input,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        rotation_link_input = ueFxUtils.create_script_input_linked_parameter(
            "Particles.MeshRotationEuler",
            ue.NiagaraScriptInputType.VEC3)
        
        angle_input = ueFxUtils.create_script_input_float(360)
        script_asset = ueFxUtils.create_asset_data(Paths.di_quaternion_from_euler)
        script_args = ue.CreateScriptContextArgs(script_asset)
        orientation_script = ueFxUtils.create_script_context(script_args)
        orientation_script.set_parameter("Euler", rotation_link_input)
        orientation_script.set_parameter("Scale", angle_input)
        orientation_input = ueFxUtils.create_script_input_dynamic(
            orientation_script,
            ue.NiagaraScriptInputType.QUATERNION)

        # --- porting from UE5.3 begin ---
        if b_inherit_parent_rotation:
            # if inheriting parent rotation, multiply the owning component 
            # orientation quat with the desired particle orientation quat
            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_quaternions)
            script_args = ue.CreateScriptContextArgs(script_asset)
            multiply_quat_script = ueFxUtils.create_script_context(script_args)

            # make and set the input for owner orientation, and set the input
            # for particle orientation
            owner_orientation_input = ueFxUtils.create_script_input_linked_parameter(
                "Engine.Owner.Rotation",
                ue.NiagaraScriptInputType.QUATERNION)
            multiply_quat_script.set_parameter("Quaternion A", orientation_input)
            multiply_quat_script.set_parameter("Quaternion B", owner_orientation_input)

            orientation_input = ueFxUtils.create_script_input_dynamic(
                multiply_quat_script,
                ue.NiagaraScriptInputType.QUATERNION)

        # set the desired rotation
        emitter.set_parameter_directly(
            "Particles.MeshOrientation",
            orientation_input,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)
        # --- porting from UE5.3 end ---
