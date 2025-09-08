# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths
from ModuleConversionScripts import CascadeMeshRotationRateToNiagara


class CascadeMeshRotationRateOverLifeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleMeshRotationRateOverLife

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        (rot_rate_dist, scale_rot_rate) = ueFxUtils.get_particle_module_mesh_rotation_rate_over_life_props(cascade_module)

        mesh_rotation_rate_module = args.find_cascade_module(ue.ParticleModuleMeshRotationRate)
        if mesh_rotation_rate_module is None:
            CascadeMeshRotationRateToNiagara.CascadeMeshRotationRateConverter.create_modules(args, emitter)

        if scale_rot_rate:
            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
            script_args = ue.CreateScriptContextArgs(script_asset)
            multiply_vector_script = ueFxUtils.create_script_context(script_args)

            rotation_rate_link_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.MeshRotationRate",
                ue.NiagaraScriptInputType.VEC3)
            multiply_vector_script.set_parameter('A', rotation_rate_link_input)

            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
            script_args = ue.CreateScriptContextArgs(script_asset)
            inner_multiply_vector_script = ueFxUtils.create_script_context(script_args)

            rot_rate_input = c2nUtils.create_script_input_for_distribution(rot_rate_dist)
            inner_multiply_vector_script.set_parameter('A', rot_rate_input)
            inner_multiply_vector_script.set_parameter('B', ueFxUtils.create_script_input_vector(ue.Vector(360.0, 360.0, 360.0)))

            inner_vector_input = ueFxUtils.create_script_input_dynamic(inner_multiply_vector_script, ue.NiagaraScriptInputType.VEC3)
            multiply_vector_script.set_parameter('B', inner_vector_input)

            result_vector_input = ueFxUtils.create_script_input_dynamic(multiply_vector_script, ue.NiagaraScriptInputType.VEC3)

            emitter.set_parameter_directly(
                "Particles.MeshRotationRate",
                result_vector_input,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)
        else:
            script_asset = ueFxUtils.create_asset_data(Paths.di_add_vector)
            script_args = ue.CreateScriptContextArgs(script_asset)
            add_vector_script = ueFxUtils.create_script_context(script_args)

            rotation_rate_link_input = ueFxUtils.create_script_input_linked_parameter(
                "Particles.MeshRotationRate",
                ue.NiagaraScriptInputType.VEC3)
            add_vector_script.set_parameter('A', rotation_rate_link_input)

            rot_rate_input = c2nUtils.create_script_input_for_distribution(rot_rate_dist)
            add_vector_script.set_parameter('B', rot_rate_input)

            result_vector_input = ueFxUtils.create_script_input_dynamic(add_vector_script, ue.NiagaraScriptInputType.VEC3)

            emitter.set_parameter_directly(
                "Particles.MeshRotationRate",
                result_vector_input,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)
