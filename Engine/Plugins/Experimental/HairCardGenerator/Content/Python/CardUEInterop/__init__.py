# Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

import unreal

@unreal.uclass()
class HairCardGenForwardingController(unreal.HairCardGenControllerBase):
    def __init__(self):
        super().__init__()
        self._wrapped_inst = None

    def _init_wrapped_controller(self):
        from .ue_card_gen_controller import HairCardGenController
        if self._wrapped_inst is None:
            self._wrapped_inst = HairCardGenController()

    @unreal.ufunction(override=True)
    def load_settings(self, generator_settings) -> bool:
        self._init_wrapped_controller()
        return self._wrapped_inst.load_settings(generator_settings)

    @unreal.ufunction(override=True)
    def get_points_per_curve(self) -> int:
        self._init_wrapped_controller()
        return self._wrapped_inst.get_points_per_curve()

    @unreal.ufunction(override=True)
    def load_groom_data(self, groom_data, name, cached_grooms_path, save_cached = False) -> bool:
        self._init_wrapped_controller()
        return self._wrapped_inst.load_groom_data(groom_data, name, cached_grooms_path, save_cached)

    @unreal.ufunction(override=True)
    def generate_clumps(self, settings_group_index: int) -> unreal.Array(int):
        self._init_wrapped_controller()
        return self._wrapped_inst.generate_clumps(settings_group_index)

    @unreal.ufunction(override=True)
    def set_optimizations(self, settings_group_index: int) -> unreal.Array(int):
        self._init_wrapped_controller()
        return self._wrapped_inst.set_optimizations(settings_group_index)

    @unreal.ufunction(override=True)
    def set_interpolated_avg_curve(self, id: int, cid: int, points):
        self._init_wrapped_controller()
        return self._wrapped_inst.set_interpolated_avg_curve(id, cid, points)

    @unreal.ufunction(override=True)
    def get_average_curve(self, id: int, cid: int) -> unreal.Array(float):
        self._init_wrapped_controller()
        return self._wrapped_inst.get_average_curve(id, cid)

    @unreal.ufunction(override=True)
    def generate_cards_geometry(self) -> unreal.Array(float):
        self._init_wrapped_controller()
        return self._wrapped_inst.generate_cards_geometry()

    @unreal.ufunction(override=True)
    def cluster_textures(self, settings_group_index: int):
        self._init_wrapped_controller()
        return self._wrapped_inst.cluster_textures(settings_group_index)

    @unreal.ufunction(override=True)
    def generate_texture_layout(self):
        self._init_wrapped_controller()
        return self._wrapped_inst.generate_texture_layout()

    @unreal.ufunction(override=True)
    def generate_texture_atlases(self, width_scale = -1):
        self._init_wrapped_controller()
        return self._wrapped_inst.generate_texture_atlases(width_scale)

    @unreal.ufunction(override=True)
    def generate_mesh(self, output_mesh: unreal.StaticMesh):
        self._init_wrapped_controller()
        return self._wrapped_inst.generate_mesh(output_mesh)


def instantiate_card_gen_controller():
    global hair_card_controller_inst
    hair_card_controller_inst = HairCardGenForwardingController()
