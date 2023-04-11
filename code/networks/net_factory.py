from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, UNet_CCT, UNet_CCT_3H, UNet_Head, UNet_MultiHead, UNet_LC, UNet_LC_MultiHead,UNet_LC_MultiHead_Two


def net_factory(args, net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_3h":
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "unet_head":
        net = UNet_Head(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_multihead":
        net = UNet_MultiHead(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_lc":
        net = UNet_LC(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid).cuda()
    elif net_type == "unet_lc_multihead":
        net = UNet_LC_MultiHead(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid).cuda()
    else:
        net = None
    return net
