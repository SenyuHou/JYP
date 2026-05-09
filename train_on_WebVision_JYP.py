import tqdm
import random
import time
import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision
from utils.ema import EMA
from utils.vit_wrapper import vit_img_wrap
from utils.animal_data_utils import Animal10N_dataset
from utils.webvision_data_utils import WebVision_dataset
from utils.model_ResNet import CustomResNetEncoder, pretrain_resnet
from utils.model_SimCLR import SimCLR_encoder
from utils.plot_loss import plot_and_save_losses, plot_historical_difference
import torch.optim as optim
from utils.learning import *
from utils.precorrct_labels import *
from utils.ws_augmentation import *
from utils.model_diffusion import Diffusion
from utils.log_config import setup_logger
from utils.add_cifar_noise import generate_instance_noise_labels, generate_instance_dependent_noise, add_noise


# Main training function
def train(diffusion_model, train_dataset_all, train_embed_dir, test_dataset, test_embed_dir, model_path, args, fp_dim):

    """
    Train the diffusion model with the given datasets and arguments.
    """
    print(f'Use cosine: {args.use_cos}, Use loss weights: {args.loss_w}, Use Single label: {args.to_single_label}, Use One view: {args.one_view}, Beta: {args.BETA}')
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    class_size = args.cls_size
    n_epochs = args.nepoch
    k = args.k
    warmup_epochs = args.warmup_epochs
    historical_epochs = args.historical_epochs
    clean_t = args.clean_t
    noisy_t = args.noisy_t
    Beta = args.BETA
    Gamma = args.GAMMA
    seed = args.seed
    train_dataset = train_dataset_all

    noisy_labels = torch.tensor(train_dataset.targets).to(torch.long).clone().squeeze().to(device)
    noisy_labels_all = torch.tensor(train_dataset_all.targets).to(torch.long).clone().squeeze().to(device)

    if args.fp_encoder == 'ResNet':
        pretrain_dataset = train_dataset
        pretest_dataset = test_dataset
        pretrain_resnet(
            diffusion_model.fp_encoder,
            pretrain_dataset,
            pretest_dataset,
            device=args.device,
            num_epochs=50,
            batch_size=args.batch_size,
            learning_rate=5e-2
        )
    diffusion_model.fp_encoder.eval()

    print('Doing pre-computing fp embeddings for weak and strong dataset')
    weak_embed, strong_embed = prepare_2_fp_x(
        args, diffusion_model.fp_encoder, train_dataset_all,
        save_dir=train_embed_dir, device=device, fp_dim=fp_dim
    )
    weak_embed = weak_embed.to(device)
    strong_embed = strong_embed.to(device)

    print('Doing pre-computing fp embeddings for test dataset')
    test_embed = prepare_fp_x(args, diffusion_model.fp_encoder, test_dataset, test_embed_dir, device=device, fp_dim=fp_dim)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=init_fn
    )
    test_loader = data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    mse_loss = nn.MSELoss(reduction='none')

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    max_accuracy_top1 = 0.0
    max_accuracy_top5 = 0.0
    historical_diff_w = torch.zeros(len(train_dataset.targets)).to(device)
    historical_diff_s = torch.zeros(len(train_dataset.targets)).to(device)
    losses = []
    print('Diffusion training start')

    for epoch in range(n_epochs):
        diffusion_model.diffusion_encoder.train()
        diffusion_model.model.train()
        total_loss = 0.0
        total_batches = 0
        epoch_diff_w = torch.zeros(len(train_dataset.targets)).to(device)
        epoch_diff_s = torch.zeros(len(train_dataset.targets)).to(device)

        if epoch >= historical_epochs:
            if not args.one_view:
                clean_prob_w, noisy_prob_w, predict_clean_id_w, predict_noisy_id_w, difficulty_w = fit_gmm(
                    historical_diff_w, noisy_labels, n_class, clean_t, noisy_t, args.gmm_by_class
                )
                clean_prob_s, noisy_prob_s, predict_clean_id_s, predict_noisy_id_s, difficulty_s = fit_gmm(
                    historical_diff_s, noisy_labels, n_class, clean_t, noisy_t, args.gmm_by_class
                )

                predict_clean_id = list(set(predict_clean_id_w).intersection(set(predict_clean_id_s)))
                predict_noisy_id = list(set(predict_noisy_id_w).intersection(set(predict_noisy_id_s)))
                predict_hard_id = list(set(predict_clean_id_w + predict_noisy_id_w) - set(predict_clean_id) - set(predict_noisy_id))
                logger.info(f'Epoch {epoch} —— Clean samples: {len(predict_clean_id)}, Noisy samples: {len(predict_noisy_id)}, Hard samples: {len(predict_hard_id)}')

            else:
                clean_prob_w, noisy_prob_w, predict_clean_id_w, predict_noisy_id_w, predict_hard_id_w, difficulty_w = fit_gmm(
                    historical_diff_w, noisy_labels, n_class, clean_t, noisy_t, args.gmm_by_class
                )
                predict_clean_id = predict_clean_id_w
                predict_noisy_id = predict_noisy_id_w
                predict_hard_id = predict_hard_id_w

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch_w, x_batch_s, y_batch, data_indices] = data_batch[:4]
                x_batch_w = x_batch_w.to(device)
                x_batch_s = x_batch_s.to(device)
                y_label_batch_noisy = y_batch.to(device)
                data_indices = data_indices.to(device)
                y_label_batch_noisy_onehot = nn.functional.one_hot(y_label_batch_noisy, num_classes=n_class).float()

                fp_embd_w = weak_embed[data_indices, :].to(device)
                fp_embd_s = strong_embed[data_indices, :].to(device)

                y_label_batch_w, y_label_batch_s, loss_weights_w, loss_weights_s = sample_labels_in_two_view(
                    fp_embd_w=fp_embd_w, fp_embd_s=fp_embd_s, y_noisy=y_label_batch_noisy,
                    weak_embed=weak_embed, strong_embed=strong_embed, noisy_labels=noisy_labels_all,
                    k=k, n_class=n_class, use_cosine_similarity=args.use_cos,
                    to_single_label=args.to_single_label, device=device
                )

                y_0_batch_w = y_label_batch_w.to(device)
                y_0_batch_s = y_label_batch_s.to(device)

                adjust_learning_rate(
                    optimizer, i / len(train_loader) + epoch,
                    warmup_epochs=warmup_epochs, n_epochs=n_epochs, lr_input=args.lr
                )

                t1 = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(x_batch_w.size(0) // 2 + 1, )).to(device)
                t1 = torch.cat([t1, diffusion_model.num_timesteps - 1 - t1], dim=0)[:x_batch_w.size(0)]
                t2 = (t1 * Gamma).to(torch.int64)

                if not args.one_view:
                    output_w, e_w = diffusion_model.forward_t(y_0_batch_w, x_batch_w, t1, fp_embd_w)
                    output_s, e_s = diffusion_model.forward_t(y_0_batch_s, x_batch_s, t2, fp_embd_s)

                    prob_w = F.softmax(output_w, dim=1).clamp(min=1e-7, max=1.0)
                    prob_s = F.softmax(output_s, dim=1).clamp(min=1e-7, max=1.0)

                    y_noisy_idx = y_label_batch_noisy_onehot.argmax(dim=1)

                    distance_w = 1.0 - prob_w.gather(1, y_noisy_idx.view(-1, 1)).squeeze(1)
                    distance_s = 1.0 - prob_s.gather(1, y_noisy_idx.view(-1, 1)).squeeze(1)

                    adjusted_distance_w = adjust_distance_by_t(distance_w, t1, diffusion_model.num_timesteps)
                    adjusted_distance_s = adjust_distance_by_t(distance_s, t2, diffusion_model.num_timesteps)

                    epoch_diff_w[data_indices] += adjusted_distance_w
                    epoch_diff_s[data_indices] += adjusted_distance_s

                    if epoch >= historical_epochs:
                        loss_w = torch.zeros(x_batch_w.size(0), device=device)
                        loss_s = torch.zeros(x_batch_s.size(0), device=device)

                        batch_idx = torch.arange(x_batch_w.size(0), device=device)
                        data_indices_batch = data_indices[batch_idx]

                        clean_mask = torch.isin(data_indices_batch, torch.tensor(predict_clean_id, device=device))
                        noisy_mask = torch.isin(data_indices_batch, torch.tensor(predict_noisy_id, device=device))
                        hard_mask = torch.isin(data_indices_batch, torch.tensor(predict_hard_id, device=device))

                        clean_batch_idx = batch_idx[clean_mask].clone().detach()
                        clean_idx = data_indices_batch[clean_mask].clone().detach()
                        noisy_batch_idx = batch_idx[noisy_mask].clone().detach()
                        noisy_idx = data_indices_batch[noisy_mask].clone().detach()
                        hard_batch_idx = batch_idx[hard_mask].clone().detach()
                        hard_idx = data_indices_batch[hard_mask].clone().detach()

                        if len(noisy_batch_idx) > 0:
                            corrected_label_w = noisy_prob_w[noisy_idx].unsqueeze(1) * prob_w[noisy_batch_idx] + (1 - noisy_prob_w[noisy_idx].unsqueeze(1)) * y_0_batch_w[noisy_batch_idx]
                            corrected_label_s = noisy_prob_s[noisy_idx].unsqueeze(1) * prob_s[noisy_batch_idx] + (1 - noisy_prob_s[noisy_idx].unsqueeze(1)) * y_0_batch_s[noisy_batch_idx]
                            corrected_label_w = sharpen_labels(corrected_label_w)
                            corrected_label_s = sharpen_labels(corrected_label_s)
                            loss_w[noisy_batch_idx] = ce_loss(prob_w[noisy_batch_idx], corrected_label_w)
                            loss_s[noisy_batch_idx] = ce_loss(prob_s[noisy_batch_idx], corrected_label_s)

                        if len(clean_batch_idx) > 0:
                            loss_w[clean_batch_idx] = ce_loss(prob_w[clean_batch_idx], y_0_batch_w[clean_batch_idx])
                            loss_s[clean_batch_idx] = ce_loss(prob_s[clean_batch_idx], y_0_batch_s[clean_batch_idx])

                        if len(hard_batch_idx) > 0:
                            loss_w[hard_batch_idx] = gce_loss(prob_w[hard_batch_idx], prob_s[hard_batch_idx], y_0_batch_w[hard_batch_idx])
                            loss_s[hard_batch_idx] = loss_w[hard_batch_idx]

                        loss_w_s = mse_loss(output_w, output_s)
                        loss_w_s = ((1 - Beta) * difficulty_w[data_indices].unsqueeze(1) + Beta * difficulty_s[data_indices].unsqueeze(1)) * loss_w_s

                    else:
                        loss_w = ce_loss(output_w, y_0_batch_w)
                        loss_s = ce_loss(output_s, y_0_batch_s)
                        loss_w_s = mse_loss(output_w, output_s)

                    if args.loss_w:
                        weighted_loss_w = torch.matmul(loss_weights_w, loss_w)
                        weighted_loss_s = torch.matmul(loss_weights_s, loss_s)
                        loss_weights_w_s = (1 - Beta) * loss_weights_w + Beta * loss_weights_s
                        weighted_loss_w_s = torch.matmul(loss_weights_w_s, loss_w_s)
                        weighted_loss = (1 - Beta) * weighted_loss_w + Beta * weighted_loss_s + Gamma * weighted_loss_w_s.mean()
                    else:
                        weighted_loss = (1 - Beta) * loss_w + Beta * loss_s + Gamma * loss_w_s.mean()

                else:
                    output_w_t1, e_w_t1 = diffusion_model.forward_t(y_0_batch_w, x_batch_w, t1, fp_embd_w)
                    output_w_t2, e_w_t2 = diffusion_model.forward_t(y_0_batch_w, x_batch_w, t2, fp_embd_w)

                    ce_loss_w_t1 = ce_loss(output_w_t1, y_0_batch_w)
                    ce_loss_w_t2 = ce_loss(output_w_t2, y_0_batch_w)
                    mes_loss_w_t1t2 = mse_loss(output_w_t1, output_w_t2)

                    prob_w = F.softmax(output_w_t1, dim=1)
                    y_noisy_idx = y_label_batch_noisy_onehot.argmax(dim=1)
                    distance_w = 1 - prob_w.gather(1, y_noisy_idx.view(-1, 1)).squeeze(1)
                    adjusted_distance_w = adjust_distance_by_t(distance_w, t1, diffusion_model.num_timesteps)

                    if args.loss_w:
                        weighted_ce_loss_w_t1 = torch.matmul(loss_weights_w, ce_loss_w_t1)
                        weighted_ce_loss_w_t2 = torch.matmul(loss_weights_w, ce_loss_w_t2)
                        weighted_mes_loss_w_t1t2 = torch.matmul(loss_weights_w, mes_loss_w_t1t2)
                    else:
                        weighted_ce_loss_w_t1 = ce_loss_w_t1
                        weighted_ce_loss_w_t2 = ce_loss_w_t2
                        weighted_mes_loss_w_t1t2 = mes_loss_w_t1t2

                    epoch_diff_w[data_indices] += adjusted_distance_w
                    weighted_loss = weighted_ce_loss_w_t1 + weighted_ce_loss_w_t2 + Gamma * weighted_mes_loss_w_t1t2.mean()

                loss = torch.mean(weighted_loss)
                pbar.set_postfix({'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

        historical_diff_w += epoch_diff_w
        historical_diff_s += epoch_diff_s

        if epoch >= warmup_epochs:
            test_acc_top1, test_acc_top5 = test(diffusion_model, test_loader, test_embed)
            logger.info(f"epoch: {epoch}, Top-1 test accuracy: {test_acc_top1:.3f}%, best Top-1 test accuracy: {max_accuracy_top1:.3f}%")
            logger.info(f"epoch: {epoch}, Top-5 test accuracy: {test_acc_top5:.3f}%, best Top-5 test accuracy: {max_accuracy_top5:.3f}%")
            if test_acc_top1 > max_accuracy_top1:
                if args.device is None:
                    states = [diffusion_model.model.module.state_dict(), diffusion_model.diffusion_encoder.module.state_dict()]
                else:
                    states = [diffusion_model.model.state_dict(), diffusion_model.diffusion_encoder.state_dict()]
                torch.save(states, model_path)
                max_accuracy_top1 = max(max_accuracy_top1, test_acc_top1)
                logger.info(f"Improved! Model saved, update best Top-1 accuracy at Epoch {epoch}, best Top-1 test accuracy: {max_accuracy_top1:.3f}")
            if test_acc_top5 > max_accuracy_top5:
                max_accuracy_top5 = max(max_accuracy_top5, test_acc_top5)
                logger.info(f"Improved! Update best Top-5 accuracy at Epoch {epoch}, best Top-5 test accuracy: {max_accuracy_top5:.3f}")


def test(diffusion_model, test_loader, test_embed):
    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt_top1 = 0
        correct_cnt_top5 = 0
        all_cnt = 0
        for idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Doing DDIM...', ncols=100):
            [images, target, indicies] = data_batch[:3]
            target = target.to(device)

            fp_embed = test_embed[indicies, :].to(device)
            output_t_0 = diffusion_model.reverse_predict(images, fp_x=fp_embed).detach().cpu()
            label_t_0 = F.softmax(output_t_0, dim=1)
            correct_1_5 = cnt_agree(label_t_0.detach().cpu(), target.cpu(), topk=(1, 5), softmax=False)
            correct_cnt_top1 += correct_1_5[1]
            correct_cnt_top5 += correct_1_5[5]
            all_cnt += images.shape[0]

    acc_top1 = 100 * correct_cnt_top1 / all_cnt
    acc_top5 = 100 * correct_cnt_top5 / all_cnt
    return acc_top1, acc_top5


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--dataset_path', default='../DATASETS/', help='dataset path')
    parser.add_argument('--dataset', default='webvision', choices=['animal10n', 'webvision'], help='dataset')
    parser.add_argument("--device", default=None, help="which GPU to use", type=str)
    parser.add_argument("--gpu_devices", default=None, type=int, nargs='+', help="")

    parser.add_argument("--nepoch", default=300, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=128, help="batch_size", type=int)
    parser.add_argument("--num_workers", default=16, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    parser.add_argument("--historical_epochs", default=100, help="historical calculate epochs", type=int)
    parser.add_argument("--cls_size", default=500, help="class size for balance", type=int)
    parser.add_argument("--lr", default=1e-3, help="learning rate", type=float)
    parser.add_argument("--BETA", default=0.2, help="loss weight for strong aug view", type=float)
    parser.add_argument("--GAMMA", default=0.2, help="loss weight for consistency mse", type=float)
    parser.add_argument("--gmm_by_class", default=False, help="fit gmm by each class", action='store_true')
    parser.add_argument("--clean_t", default=0.5, help="GMM clean threshold", type=float)
    parser.add_argument("--noisy_t", default=0.5, help="GMM noisy threshold", type=float)

    parser.add_argument("--feature_dim", default=1024, help="feature dim for encoder in diffusion model", type=int)
    parser.add_argument("--k", default=50, help="k neighbors for knn", type=int)
    parser.add_argument("--loss_w", default=False, help="use neighbor frec weights for loss", action='store_true')
    parser.add_argument("--to_single_label", default=False, help="use single label for label correction", action='store_true')
    parser.add_argument("--one_view", default=False, help="use single view for diffusion", action='store_true')
    parser.add_argument("--use_cos", default=False, help="use cosine for neighbor space", action='store_true')
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--one_step", default=False, help="one step for sampling", action='store_true')
    parser.add_argument("--diff_encoder", default='resnet50_l', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)

    parser.add_argument("--fp_encoder", default='ViT', help="which encoder for fp (SimCLR, Vit or ResNet)", type=str)
    parser.add_argument("--fp_encoder_version", default='ViT-L/14', help="which encoder for Vit", type=str)

    parser.add_argument("--plot_his_diff", default=False, help="plot and save figures for his_diff", action='store_true')
    parser.add_argument("--log_path", default='./logs/', help="input your logs path", type=str)
    parser.add_argument("--alg_name", default='JYP', help="create your alg name", type=str)
    args = parser.parse_args()

    if args.device is not None:
        dev_str = args.device.strip().lower()
        if dev_str == "cpu":
            device = torch.device("cpu")
            gpu_ids = []
        else:
            gid = int(dev_str)
            device = torch.device(f"cuda:{gid}")
            gpu_ids = [gid]
    else:
        if args.gpu_devices is None or len(args.gpu_devices) == 0:
            gpu_ids = [0]
        else:
            gpu_ids = args.gpu_devices
        device = torch.device(f"cuda:{gpu_ids[0]}")

    print(f"Main device: {device}, all GPUs: {gpu_ids}")

    print('Using device:', device)
    logger = setup_logger(args)
    dataset = args.dataset

    if dataset == 'animal10n':
        n_class = 10
        DATA_MEAN = (0.6959, 0.6537, 0.6371)
        DATA_STD = (0.3113, 0.3192, 0.3214)
        data_dir = os.path.join(args.dataset_path, 'Animal10N')
        print('Load Animal10N data from: ', data_dir)
        train_dataset = Animal10N_dataset(root_dir=data_dir, mode='train')
        test_dataset = Animal10N_dataset(root_dir=data_dir, mode='test')
        train_embed_dir = os.path.join(data_dir, f'fp_embed_train_animal_by_{args.fp_encoder}')
        test_embed_dir = os.path.join(data_dir, f'fp_embed_test_animal_by_{args.fp_encoder}')

    elif dataset == 'webvision':
        n_class = 50
        webvision_dir = os.path.join(args.dataset_path, 'WebVision')
        print('Load WebVision data from: ', webvision_dir)
        DATA_MEAN = (0.485, 0.456, 0.406)
        DATA_STD = (0.229, 0.224, 0.225)
        fp_embed_dir = os.path.join(webvision_dir, 'fp_embed_dir')
        if not os.path.exists(fp_embed_dir):
            os.makedirs(fp_embed_dir)
        if args.cls_size == 0:
            is_balance = False
            train_embed_dir = os.path.join(fp_embed_dir, f'fp_embed_train_webvision_by_{args.fp_encoder}')
        else:
            is_balance = True
            train_embed_dir = os.path.join(fp_embed_dir, f'fp_embed_train_webvision_by_{args.fp_encoder}_cls_size{args.cls_size}')
        train_dataset_all = WebVision_dataset(data_root=webvision_dir, split='train', balance=is_balance, randomize=False, cls_size=args.cls_size)
        logger.info(f"Total training samples: {len(train_dataset_all)}")
        val_dataset = WebVision_dataset(data_root=webvision_dir, split='val')
        test_embed_dir = os.path.join(fp_embed_dir, f'fp_embed_val_webvision_by_{args.fp_encoder}')

    else:
        raise Exception("Dataset should be animal10n or webvision")

    if args.fp_encoder == 'SimCLR':
        fp_dim = 2048
        state_dict = torch.load(f'./model/SimCLR_128_{dataset}.pt', map_location=device)
        fp_encoder = SimCLR_encoder(feature_dim=128).to(device)
        fp_encoder.load_state_dict(state_dict, strict=False)
    elif args.fp_encoder == 'ViT':
        fp_encoder = vit_img_wrap(args.fp_encoder_version, device, center=DATA_MEAN, std=DATA_STD)
        fp_dim = fp_encoder.dim
    elif args.fp_encoder == 'ResNet':
        fp_encoder = CustomResNetEncoder(base_model=args.fp_encoder_version, num_class=n_class).to(device)
        fp_dim = fp_encoder.feature_dim
    else:
        raise Exception("fp_encoder should be SimCLR, Vit or ResNet")

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_path = f'./model/{args.alg_name}_{args.fp_encoder}_{args.dataset}_{timestamp}.pt'
    diffusion_model = Diffusion(
        fp_encoder_type=args.fp_encoder, fp_encoder=fp_encoder, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim,
        device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step, beta_schedule='cosine', one_step=args.one_step
    )

    if len(gpu_ids) > 1 and device.type == "cuda":
        print(f'using DataParallel on GPUs: {gpu_ids}, main device: {device}')

        diffusion_model.model.to(device)
        diffusion_model.diffusion_encoder.to(device)
        diffusion_model.fp_encoder.to(device)

        diffusion_model.model = nn.DataParallel(diffusion_model.model, device_ids=gpu_ids)
        diffusion_model.diffusion_encoder = nn.DataParallel(diffusion_model.diffusion_encoder, device_ids=gpu_ids)
        diffusion_model.fp_encoder = nn.DataParallel(diffusion_model.fp_encoder, device_ids=gpu_ids)
    else:
        print('using single GPU or CPU')
        diffusion_model.to(device)

    print(f'Training JYP using fp encoder: {args.fp_encoder} on: {args.dataset}.')
    print(f'Model saving dir: {model_path}')
    train(
        diffusion_model,
        train_dataset_all=train_dataset_all,
        train_embed_dir=train_embed_dir,
        test_dataset=val_dataset,
        test_embed_dir=test_embed_dir,
        model_path=model_path,
        args=args,
        fp_dim=fp_dim
    )
