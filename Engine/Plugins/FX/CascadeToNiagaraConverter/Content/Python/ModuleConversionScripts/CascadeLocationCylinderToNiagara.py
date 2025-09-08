# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeCylinderLocationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLocationPrimitiveCylinder

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # find/add the module script for cylinder location
        script_asset = ueFxUtils.create_asset_data(Paths.script_location_shape)
        script_args = ue.CreateScriptContextArgs(script_asset)
        cylinder_script = emitter.find_or_add_module_script(
            "ShapeLocation",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        # get all properties from the cascade cylinder location module
        # noinspection PyTypeChecker
        (b_radial_velocity,
         start_radius_distribution,
         start_height_distribution,
         height_axis,
         b_positive_x,
         b_positive_y,
         b_positive_z,
         b_negative_x,
         b_negative_y,
         b_negative_z,
         b_surface_only,
         b_velocity,
         velocity_scale_distribution,
         start_location_distribution
         ) = ueFxUtils.get_particle_module_location_primitive_cylinder_props(cascade_module)

        # Set shape to Cylinder
        cylinder_script.set_parameter(
            "Shape Primitive",
            ueFxUtils.create_script_input_enum(Paths.enum_shape_primitive, 'Cylinder'))

        # OwnerRotation is not reflected correctly in Axis Angle, so completely rewrite as Yaw/Pitch/Roll.
        euler = ue.Vector(0.0, 0.0, 0.0) # Yaw/Pitch/Roll
        scale = ue.Vector(1.0, 1.0, 1.0)
        midpoint = 0.5
        hemicircle_x = False
        hemicircle_y = False
        if height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_X:
            euler.y = 0.25
            if b_positive_x != b_negative_x:
                midpoint = 1.0 if b_positive_x else 0.0
                scale.z = 0.5
            if b_positive_y != b_negative_y:
                hemicircle_y = True
                scale.y = 1.0 if b_positive_y else -1.0
            if b_positive_z != b_negative_z:
                hemicircle_x = True
                scale.x = 1.0 if b_positive_z else -1.0
        elif height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_Y:
            euler.z = 0.25
            if b_positive_x != b_negative_x:
                hemicircle_x = True
                scale.x = 1.0 if b_positive_x else -1.0
            if b_positive_y != b_negative_y:
                midpoint = 0.0 if b_positive_y else 1.0
                scale.z = 0.5
            if b_positive_z != b_negative_z:
                hemicircle_y = True
                scale.y = -1.0 if b_positive_z else 1.0
        elif height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_Z:
            if b_positive_x != b_negative_x:
                hemicircle_x = True
                scale.x = 1.0 if b_positive_x else -1.0
            if b_positive_y != b_negative_y:
                hemicircle_y = True
                scale.y = 1.0 if b_positive_y else -1.0
            if b_positive_z != b_negative_z:
                midpoint = 0.0 if b_positive_z else 1.0
                scale.z = 0.5
        else:
            raise NameError("Failed to get valid height axis from cylinder "
                            "location module!")

        options = c2nUtils.DistributionConversionOptions()
        options.set_index_by_emitter_age()
        radius_input = c2nUtils.create_script_input_for_distribution(start_radius_distribution, options)
        cylinder_script.set_parameter("Cylinder Radius", radius_input)

        height_input = c2nUtils.create_script_input_for_distribution(start_height_distribution, options)
        cylinder_script.set_parameter("Cylinder Height", height_input)

        midpoint_input = ueFxUtils.create_script_input_float(midpoint)
        cylinder_script.set_parameter("Cylinder Height Midpoint", midpoint_input)

        cylinder_script.set_parameter("Hemicircle X", ueFxUtils.create_script_input_bool(hemicircle_x))
        cylinder_script.set_parameter("Hemicircle Y", ueFxUtils.create_script_input_bool(hemicircle_y))

        cylinder_script.set_parameter("Non Uniform Scale", ueFxUtils.create_script_input_vector(scale))
        cylinder_script.set_parameter("Rotation Mode", ueFxUtils.create_script_input_enum(Paths.enum_rotation_mode, 'Yaw / Pitch / Roll'))
        cylinder_script.set_parameter("Yaw / Pitch / Roll", ueFxUtils.create_script_input_vector(euler))

        # set surface only emission if required.
        if b_surface_only:
            cylinder_script.set_parameter(
                "Surface Only Band Thickness",
                ueFxUtils.create_script_input_float(0.0),
                True,
                True)
            # For some reason it doesn't work, but I'll put it in anyway.
            cylinder_script.set_parameter("Use Endcaps In Surface Only Mode", ueFxUtils.create_script_input_bool(False))
            cylinder_script.set_parameter("Hemicircle Internal Cap", ueFxUtils.create_script_input_bool(False))

        fix_shape_vector_script = None

        # add velocity along the cylinder if required.
        if b_velocity:
            # Add ShapeVector modification module
            script_asset = ueFxUtils.create_asset_data(Paths.script_fix_shape_vector)
            script_args = ue.CreateScriptContextArgs(script_asset)
            fix_shape_vector_script = emitter.find_or_add_module_script(
                "FixShapeVector",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)
            
            # add a script to add velocity.
            script_asset = ueFxUtils.create_asset_data(Paths.script_add_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset, [1, 2])
            add_velocity_script = emitter.find_or_add_module_script(
                "AddCylinderVelocity",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            velocity_input = ueFxUtils.create_script_input_linked_parameter(
                "Output.FixShapeVector.ShapeVector",
                ue.NiagaraScriptInputType.VEC3)

            # if radial velocity is specified, zero the velocity component on 
            # the cylinder up vector.
            if b_radial_velocity:
                if height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_X:
                    vector_mask_input = ueFxUtils.create_script_input_vector(ue.Vector(0.0, 1.0, 1.0))
                elif height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_Y:
                    vector_mask_input = ueFxUtils.create_script_input_vector(ue.Vector(1.0, 0.0, 1.0))
                elif height_axis == ue.CylinderHeightAxis.PMLPC_HEIGHTAXIS_Z:
                    vector_mask_input = ueFxUtils.create_script_input_vector(ue.Vector(1.0, 1.0, 0.0))
                else:
                    raise NameError("Failed to get valid height axis from cylinder location module!")

                # mask the configured component of the velocity.
                script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
                script_args = ue.CreateScriptContextArgs(script_asset)
                multiply_vector_script = ueFxUtils.create_script_context(script_args)
                multiply_vector_script.set_parameter("A", velocity_input)
                multiply_vector_script.set_parameter("B", vector_mask_input)
                velocity_input = ueFxUtils.create_script_input_dynamic(
                    multiply_vector_script,
                    ue.NiagaraScriptInputType.VEC3)

            # if there is velocity scaling, apply it.
            if c2nUtils.distribution_always_equals(velocity_scale_distribution, 0.0) is False:
                # make an input to calculate the velocity scale and index the 
                # scale by the emitter age.
                options = c2nUtils.DistributionConversionOptions()
                emitter_age_index = ueFxUtils.create_script_input_linked_parameter(
                    "Emitter.LoopedAge",
                    ue.NiagaraScriptInputType.FLOAT)
                options.set_custom_indexer(emitter_age_index)
                velocity_scale_input = c2nUtils.create_script_input_for_distribution(
                    velocity_scale_distribution,
                    options)

                # multiply the velocity by the scale.
                script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector_by_float)
                script_args = ue.CreateScriptContextArgs(script_asset)
                mul_vec3_by_float_script = ueFxUtils.create_script_context(script_args)
                mul_vec3_by_float_script.set_parameter("Vector", velocity_input)
                mul_vec3_by_float_script.set_parameter("Float", velocity_scale_input)
                velocity_input = ueFxUtils.create_script_input_dynamic(
                    mul_vec3_by_float_script,
                    ue.NiagaraScriptInputType.VEC3)

            # apply the velocity.
            add_velocity_script.set_parameter("Velocity", velocity_input)

            # make sure we have a solve forces and velocity script.
            script_asset = ueFxUtils.create_asset_data(Paths.script_solve_forces_and_velocity)
            script_args = ue.CreateScriptContextArgs(script_asset)
            emitter.find_or_add_module_script(
                "SolveForcesAndVelocity",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # offset the location if required.
        if c2nUtils.distribution_always_equals(start_location_distribution, 0.0) is False:
            # create an input to set the offset and index the value by emitter age.
            options = c2nUtils.DistributionConversionOptions()
            emitter_age_index = ueFxUtils.create_script_input_linked_parameter(
                "Emitter.LoopedAge",
                ue.NiagaraScriptInputType.FLOAT)
            options.set_custom_indexer(emitter_age_index)
            start_location_input = c2nUtils.create_script_input_for_distribution(start_location_distribution, options)

            cylinder_script.set_parameter(
                "Offset Mode",
                ueFxUtils.create_script_input_enum(Paths.enum_offset_mode, 'Default'))

            emitter.set_parameter_directly(
                "Particles.ShapeLocationOffset",
                start_location_input,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            shape_location_param_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.ShapeLocationOffset",
                ue.NiagaraScriptInputType.VEC3)

            cylinder_script.set_parameter("Offset", shape_location_param_input)

            coordinate_local_input = ueFxUtils.create_script_input_enum(Paths.enum_niagara_coordinate_space, "Local")

            emitter.set_parameter_directly(
                "Particles.ShapeLocationOffsetSpace",
                coordinate_local_input,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            shape_location_space_param_input = ueFxUtils.create_script_input_linked_parameter_ex(
                "Particles.ShapeLocationOffsetSpace",
                coordinate_local_input)

            cylinder_script.set_parameter(
                "Offset Coordinate Space",
                shape_location_space_param_input)
            
            if fix_shape_vector_script:
                fix_shape_vector_script.set_parameter(
                    "Offset Mode",
                    ueFxUtils.create_script_input_enum(Paths.enum_offset_mode, 'Default'))

                shape_location_param_input = ueFxUtils.create_script_input_linked_parameter(
                    "Particles.ShapeLocationOffset",
                    ue.NiagaraScriptInputType.VEC3)
                fix_shape_vector_script.set_parameter("Offset", shape_location_param_input)

                shape_location_space_param_input = ueFxUtils.create_script_input_linked_parameter_ex(
                    "Particles.ShapeLocationOffsetSpace",
                    coordinate_local_input)

                fix_shape_vector_script.set_parameter(
                    "Offset Coordinate Space",
                    shape_location_space_param_input)
