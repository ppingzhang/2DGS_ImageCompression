
from utils import *


def load_img(args, ratio = 1):
    original_image = Image.open(args.image_dir)

    original_image_or = original_image.convert('RGB')
    original_image_or = np.array(original_image_or)
    original_image_or = original_image_or / 255.0
    or_height, or_width,  _ = original_image_or.shape

    or_image_array = original_image_or
    or_target_tensor = torch.tensor(or_image_array, dtype=torch.float32, device=args.device)

    original_image_re = original_image.resize((or_width//ratio, or_height//ratio))
    original_image_re = original_image_re.convert('RGB')
    original_image_re = np.array(original_image_re)
    original_image_re = original_image_re / 255.0
    height, width,  _ = original_image_re.shape

    re_image_array = original_image_re
    target_tensor = torch.tensor(re_image_array, dtype=torch.float32, device=args.device)


    data_dict = {'img': target_tensor, 'or_img': or_target_tensor, 'height':height, 'width':width}
    return data_dict

