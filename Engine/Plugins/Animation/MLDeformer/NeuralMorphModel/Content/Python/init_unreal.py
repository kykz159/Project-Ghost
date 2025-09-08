# -*- coding: utf-8 -*-
"""
ML Deformer Neural Morph Model.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from importlib import reload
import unreal

if unreal.is_editor():
    @unreal.uclass()
    class NeuralMorphPythonTrainingModel(unreal.NeuralMorphTrainingModel):
        @unreal.ufunction(override=True)
        def update_available_devices(self):
            import mldeformer.training_helpers
            reload(mldeformer.training_helpers)
            mldeformer.training_helpers.update_training_device_list(self)

        @unreal.ufunction(override=True)
        def train(self):
            print('Training the Neural Morph Model')
            if self.get_model().mode == 0:
                import neuralmorphmodel_local
                reload(neuralmorphmodel_local)
                return neuralmorphmodel_local.train(self)
            else:
                assert self.get_model().mode == 1
                import neuralmorphmodel_global
                reload(neuralmorphmodel_global)
                return neuralmorphmodel_global.train(self)
