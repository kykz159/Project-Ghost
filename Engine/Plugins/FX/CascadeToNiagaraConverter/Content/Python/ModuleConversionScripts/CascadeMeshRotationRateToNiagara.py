# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeMeshRotationRateConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleMeshRotationRate

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties of the mesh rotation rate module.
        # noinspection PyTypeChecker
        start_rotation_rate_distribution = ueFxUtils.get_particle_module_mesh_rotation_rate_props(cascade_module)
        
        cls.create_modules(args, emitter, start_rotation_rate_distribution)

    @classmethod
    def create_modules(cls, args, emitter, start_rotation_rate_distribution=None):
        script_asset = ueFxUtils.create_asset_data(Paths.script_update_mesh_orientation_by_euler)
        script_args = ue.CreateScriptContextArgs(script_asset)
        mesh_orient_script = emitter.find_or_add_module_script(
            "UpdateMeshOrientationEuler",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        if start_rotation_rate_distribution:
            rotation_rate_base_input = c2nUtils.create_script_input_for_distribution(start_rotation_rate_distribution)
            emitter.set_parameter_directly(
                "Particles.MeshRotationRateBase",
                rotation_rate_base_input,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            rotation_rate_base_link_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.MeshRotationRateBase",
                ue.NiagaraScriptInputType.VEC3)
        else:
            rotation_rate_base_link_input = ueFxUtils.create_script_input_vector(ue.Vector(0.0, 0.0, 0.0))

        emitter.set_parameter_directly(
            "Particles.MeshRotationRate",
            rotation_rate_base_link_input,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        rotation_rate_link_input = ueFxUtils.create_script_input_linked_parameter(
            "Particles.MeshRotationRate",
            ue.NiagaraScriptInputType.VEC3)

        mesh_orient_script.set_parameter("RotationRate", rotation_rate_link_input)
