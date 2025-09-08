# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeVelocityConeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleVelocityCone

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get velocity over lifetime properties.
        # noinspection PyTypeChecker
        (angle_distribution,
         velocity_distribution,
         direction,
         b_world_space,
         b_apply_owner_scale
         ) = ueFxUtils.get_particle_module_velocity_cone_props(cascade_module)

        # Convert distributions
        angle_distribution_input = c2nUtils.create_script_input_for_distribution(angle_distribution)
        velocity_distribution_input = c2nUtils.create_script_input_for_distribution(velocity_distribution)

        # Apply Owner Scale
        #if b_apply_owner_scale:
        #    script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
        #    script_args = ue.CreateScriptContextArgs(script_asset)
        #    mul_vec_script = ueFxUtils.create_script_context(script_args)
        #    
        #    owner_scale_input = ueFxUtils.create_script_input_linked_parameter("Engine.Owner.Scale", ue.NiagaraScriptInputType.FLOAT)

        #    mul_vec_script.set_parameter("A", velocity_distribution_input)
        #    mul_vec_script.set_parameter("B", owner_scale_input)

        #    velocity_distribution_input = ueFxUtils.create_script_input_dynamic(mul_vec_script, ue.NiagaraScriptInputType.VEC3)

        # Create Module
        script_asset = ueFxUtils.create_asset_data(Paths.script_add_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset)
        add_velocity_script = ueFxUtils.create_script_context(script_args)
        emitter.add_module_script(
            "AddVelocityInCone",
            add_velocity_script,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        add_velocity_script.set_parameter("Velocity Mode", ueFxUtils.create_script_input_enum(Paths.enum_niagara_velocity_mode, "In Cone"))
        add_velocity_script.set_parameter("Velocity Speed", velocity_distribution_input)
        add_velocity_script.set_parameter("Cone Angle", angle_distribution_input)
        add_velocity_script.set_parameter("Cone Axis", ueFxUtils.create_script_input_vector(direction))
        add_velocity_script.set_parameter("Cone Angle Mode", ueFxUtils.create_script_input_enum(Paths.enum_niagara_angle_input, "Normalized Angle (0-1)"))

        # Handle world vs local space (tbd)
        #if b_world_space:
        #    add_velocity_script.set_parameter(
        #        "Rotation Coordinate Space",
        #        ueFxUtils.create_script_input_enum(Paths.enum_niagara_coordinate_space, "World"))

        # Make sure we have solve module
        script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
        script_args = ue.CreateScriptContextArgs(script_asset)
        emitter.find_or_add_module_script(
            "SolveForcesAndVelocity",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
