
import os
import sys
import arcgis
import shutil
import time
import torchvision.models


if __name__ == "__main__":

    path_training_sets = r"d:\Temp\SlideMap\DeepLearningStuff\TrainingSets_MRCNN"
    path_trained_models = r"D:\Temp\SlideMap\DeepLearningStuff\TrainingModels_MRCNN"

    memory_error = "CUDA out of memory"
    backbone_model = torchvision.models.resnet50()

    for folder in os.listdir(path_training_sets):
        current_folder = os.path.join(path_training_sets, folder)
        out_folder = os.path.join(path_trained_models, folder)

        batch_size = 32
        seed = 50
        chip_size = int(folder.split("_")[1])

        finished = False

        while True:
            if(finished or batch_size <= 0):
                break
            try:
                # print(batch_size)
                print("Processing folder", folder, "with batch size", batch_size)
                if(os.path.exists(out_folder)):
                    shutil.rmtree(out_folder)

                db = arcgis.learn.prepare_data(path=current_folder,
                                               # class_mapping="",
                                               chip_size=chip_size,
                                               batch_size=batch_size,
                                               seed=seed,
                                               # dataset_type="Classified_Tiles")
                                               dataset_type="RCNN_Masks")

                m = arcgis.learn.MaskRCNN(data=db)
                # m = arcgis.learn.UnetClassifier(data=db)
                m.fit(epochs=50)
                m.save(out_folder)
                finished = True

            except Exception as e:
                print(e)

                if (memory_error in str(e)):
                    if(batch_size == 2):
                        batch_size = batch_size - 1
                    else:
                        batch_size = batch_size - 2
                    print("Decreasing batch size to ", batch_size)