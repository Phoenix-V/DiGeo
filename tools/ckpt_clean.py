import argparse
import os
import pdb
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        "--src", type=str, default="../checkpoints/voc/prior/", help="Path to the main checkpoint"
    )
    args = parser.parse_args()
    return args

def reset_ckpt(ckpt):
    if "scheduler" in ckpt:
        del ckpt["scheduler"]
    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "iteration" in ckpt:
        ckpt["iteration"] = 0


if __name__ == "__main__":
    args = parse_args()

    for exp in os.listdir(args.src):
        src_ckpt = os.path.join(args.src,exp,'model_final.pth')
        dst_ckpt = src_ckpt.replace('model_final','model_clean_student')
        if os.path.isfile(dst_ckpt):
            print('The cleaned model for distillation stage already exists for Exp {}'.format(exp))
            continue
        elif not os.path.isfile(src_ckpt):
            print('The final model is not found, please check {}'.format(exp))
            continue

        ckpt = torch.load(src_ckpt)
        reset_ckpt(ckpt)
        for key in [k for k in ckpt['model'] if 'box_' in k]:
            new_weight = ckpt['model'][key].clone()
            key_comp = key.split('.')
            new_key = '.'.join(key_comp[:1]+['student_'+key_comp[1]]+key_comp[2:])
            ckpt['model'][new_key] = new_weight

        torch.save(ckpt,dst_ckpt)
        print('The final model has been cleaned for {}'.format(exp))

