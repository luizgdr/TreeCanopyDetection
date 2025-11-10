# Tree canopy detection

- Guilherme Schwarz
- Julia Cristina Moreira da Silva
- Luiz Guilherme Durau Rodrigues
- Matheus Francisco Trevisan Del Zotto

## Files
- The saved run is at the '/saved_run' dir. 
- The trained model can be found at 'saved_run/segment/tree_seg/weights/best.pt'.
- The YOLO generated results can be found under the 'saved_run/segment/' dir.
- The raw predictions can be found under 'saved_run/segment/predict'
- The generated image predictions and masks can be found under 'saved_run/segment/generated'
- The execution logs can be found at 'saved_run/segment/logs'
- The calculated accuracy can be found on 'saved_run/segment/generated/<type>/evaluation_summary_<type>.json' where type can be yolo or sam2_deepforest

## Excecution
- Download `train_images.zip`: [Link](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?tab=&menu=data)
- Run `setup.py`
- Extract `train_images.zip` images (.tifs) to `data/images/`
- It is recommended to download CUDA on your computer.
- Run `main.py`