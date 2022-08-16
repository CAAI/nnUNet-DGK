#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict

from nnunet.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from nnunet.paths import *


class ExperimentPlannerPETasCT_2D(ExperimentPlanner2D_v21):
    """
    Preprocesses all data in CT mode
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlannerPETasCT_2D, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetPlansv2.1_PETasCT2D"
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlansv2.1_PETasCT2D" + "_plans_2D.pkl")

    def determine_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT":
                schemes[i] = "CT"
            else:
                schemes[i] = "CT"
        return schemes

