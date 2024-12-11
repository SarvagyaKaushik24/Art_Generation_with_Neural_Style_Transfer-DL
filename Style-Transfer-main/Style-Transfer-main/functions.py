from PIL import Image 
import torchvision.transforms as transforms 
import numpy as np 

image_size = 224

loader = transforms.Compose(
    [
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ]
)

def save_image(img,step) : 
    # print("Image type:", img.dtype)
    # print("Min value:", img.min())
    # print("Max value:", img.max())
    # result = Image.fromarray((img * 255).astype(np.uint8))
    # result.save(f"Generated_image_step_{step}")
    img = img.clamp(0, 1)  # Ensure values are in the range [0, 1]
    img = img.squeeze(0)   # Remove batch dimension
    img = transforms.ToPILImage()(img)  # Convert tensor to PIL image
    img.save(f"Generated_image_step_{step}.jpg")

def image_loader(img_name) : 
    image = Image.open(img_name)
    image = loader(image).unsqueeze(0)

    return image


def gram_matrix(feature, channel, height, width) : 
    G = feature.view(channel, height*width).mm(feature.view(channel, height*width).t())

    return G

