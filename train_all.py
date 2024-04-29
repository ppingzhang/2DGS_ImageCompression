import subprocess



def train():
    dict_list = {100:32, 200:64, 500:128, 700:256, 1000:256}
    
    for primary_samples in [1000]:
        for ii in range(1, 25):
            image_dir = f"./data/kodim{ii:0>2d}.png"
            for num_embeddings in [256]: # 100/32 200/32 500/52 700/128
                num_embeddings = dict_list[primary_samples]
                cmd_str = f"python main.py --primary_samples={primary_samples} --backup_samples={primary_samples} --num_embeddings={num_embeddings} --outf={primary_samples}_{num_embeddings} --num_epochs=2000 --image_dir={image_dir}"
                print(cmd_str)
                #subprocess.run(cmd_str, shell=True)


train()