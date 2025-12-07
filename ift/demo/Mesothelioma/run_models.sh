iftFLIM-ExtractFeatures  model_lungs/arch.json images_for_lung_segm  images.csv model_lungs lungs_activ 1

iftFLIM-ExtractFeatures  model_fluid/arch.json images_for_fluid_segm/  images.csv model_fluid/ fluid_activ 1

iftSkipConnection lungs_activ/ fluid_activ/ activ


