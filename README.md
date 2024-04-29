## This is the code for "2D Gaussian Splatting for Image Compression".

Training command:
> python main.py 
    --primary_samples=100 
    --backup_samples=100 
    --num_embeddings=32 
    --outf=100_32 
    --num_epochs=2000 
    --image_dir=./data/kodim01.png


| Argument | Possible values |
|------|------|
| `--primary_samples` |  the number of primary samples |
| `--backup_samples` | the number of backup samples |
| `--num_embeddings` |  the number of embeddings |
| `--outf` | the output directory  |
| `--num_epochs` | the number of epochs for training |
| `--image_dir` |  the directory of the image |

Alternatively, you can follow the settings in the train_all.py file.

During training, the best model will be saved.

Test command:
> python main.py --eval=True --primary_samples=100 --backup_samples=100 --num_embeddings=32 --outf=100_32 --num_epochs=2000 --image_dir=./data/kodim01.png 


>  **PS:**
>  - I would be very happy if you could accelerate it or convert it into CUDA language. Try it!
>  - I've exported the environment.yml file from my Conda environment for reference purposes. 
>  - Please feel free to contact me if you have any problem.


