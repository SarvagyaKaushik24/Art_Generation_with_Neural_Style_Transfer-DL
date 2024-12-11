import argparse
from NST import VGG
import functions
import torch 
import torch.optim as optim
from tqdm import tqdm 


if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description="NST")
    parser.add_argument('--style_img_pth',type=str,default="Style-Transfer-main/lightning.jpg")
    parser.add_argument('--content_img_pth',type=str,default="Style-Transfer-main/messi1.jpg")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--total_steps',type=int,default=1000)
    parser.add_argument('--learning_rate',type=float,default=0.01)
    parser.add_argument('--alpha',default=1000)
    parser.add_argument('--beta',default=0.01)
    

    args = parser.parse_args()

    print(args)

    if(torch.cuda.is_available()) : 
        device = "cuda"
    else : 
        device = "cpu"

    print(f"Device set to : {device}")

    functions.image_size = args.image_size

    content_img = functions.image_loader(args.content_img_pth).to(device)
    style_img = functions.image_loader(args.style_img_pth).to(device)

    model = VGG().to(device).eval()

    generated = content_img.clone().requires_grad_(True)

    optimizer = optim.Adam([generated], lr=args.learning_rate)

    for step in tqdm(range(args.total_steps)) : 
        generated_feat = model(generated)
        content_feat = model(content_img)
        style_feat = model(style_img)

        style_loss = 0 
        content_loss = 0 

        for gen_feat, og_feat, st_feat in zip(generated_feat, content_feat, style_feat) : 
            batch_size, channel, height, width = gen_feat.shape

            content_loss += torch.mean((gen_feat - og_feat)**2)

            # creating the content gram matrix
            G = functions.gram_matrix(gen_feat, channel, height , width)

            # creating the style gram matrix 
            S = functions.gram_matrix(st_feat, channel, height, width )

            style_loss += torch.mean((G - S)**2)

        total_loss = args.alpha*content_loss + args.beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step+1) % 200 == 0 :
            print(f"Total loss = {total_loss}")
            functions.save_image(generated,step)
            
             

        




