'''Implements a generic training loop. powerful ver
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import json


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          validation_fn=None, start_epoch=0, adjust_relative_grads=False, args=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # Weight adjustment parameter
    new_weight = 1

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    # Load the checkpoint if required
    if start_epoch > 0:
        # Load the model and start training from that point onwards
        model_path = os.path.join(model_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.train()
        optim.load_state_dict(checkpoint['optimizer'])
        optim.param_groups[0]['lr'] = lr
        assert(start_epoch == checkpoint['epoch'])
        if 'new_weight' in checkpoint.keys():
            new_weight = checkpoint['new_weight']
    else:
        # Start training from scratch
        if os.path.exists(model_dir):
            val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
            if val == 'y':
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        
    #saves input to text file
    opt_path = os.path.join(model_dir, 'commandline_args.txt')
    with open(opt_path, 'w') as f:
      json.dump(args, f, indent=2)
          

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = start_epoch

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                # Saving the optimizer state is important to produce consistent results
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'new_weight': new_weight}
                torch.save(checkpoint,
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                # torch.save(model.state_dict(),
                #            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch)

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                # Adjust the relative magnitude of the losses if required
                if adjust_relative_grads:
                    if losses['diff_constraint_hom'] > 0:
                        params = OrderedDict(model.named_parameters())
                        # Gradients with respect to the PDE loss
                        optim.zero_grad()
                        losses['diff_constraint_hom'].backward(retain_graph=True)
                        grads_PDE = []
                        for key, param in params.items():
                            grads_PDE.append(param.grad.view(-1))
                        grads_PDE = torch.cat(grads_PDE)

                        #import ipdb; ipdb.set_trace()

                        # Gradients with respect to the boundary loss
                        optim.zero_grad()
                        losses['dirichlet'].backward(retain_graph=True)
                        grads_dirichlet = []
                        for key, param in params.items():
                            grads_dirichlet.append(param.grad.view(-1))
                        grads_dirichlet = torch.cat(grads_dirichlet)

                        # # Plot the gradients
                        # import seaborn as sns
                        # import matplotlib.pyplot as plt
                        # fig = plt.figure(figsize=(5, 5))
                        # ax = fig.add_subplot(1, 1, 1)
                        # ax.set_yscale('symlog')
                        # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                        # sns.distplot(grads_dirichlet.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                        # fig.savefig('gradient_visualization.png')

                        # fig = plt.figure(figsize=(5, 5))
                        # ax = fig.add_subplot(1, 1, 1)
                        # ax.set_yscale('symlog')
                        # grads_dirichlet_normalized = grads_dirichlet * torch.mean(torch.abs(grads_PDE))/torch.mean(torch.abs(grads_dirichlet))
                        # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                        # sns.distplot(grads_dirichlet_normalized.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                        # ax.set_xlim([-1000.0, 1000.0])
                        # fig.savefig('gradient_visualization_normalized.png')

                        # Set the new weight according to the paper
                        # num = torch.max(torch.abs(grads_PDE))
                        num = torch.mean(torch.abs(grads_PDE))
                        den = torch.mean(torch.abs(grads_dirichlet))
                        new_weight = 0.9*new_weight + 0.1*num/den
                        losses['dirichlet'] = new_weight*losses['dirichlet']
                    writer.add_scalar('weight_scaling', new_weight, total_steps)

                # import ipdb; ipdb.set_trace()

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    if loss_name == 'dirichlet':
                        writer.add_scalar(loss_name, single_loss/new_weight, total_steps)
                    else:
                        writer.add_scalar(loss_name, single_loss, total_steps)

                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        device = 'cuda'
                        print("Running validation set...")
                        model.eval()
                        val_losses = []
                        
                        # Use DataParallel if multiple GPUs are available
                        if torch.cuda.device_count() > 1:
                            print(f"Using {torch.cuda.device_count()} GPUs for parallel validation...")
                            model = torch.nn.DataParallel(model)

                        # Move model to the specified device
                        model.to(device)

                        # Perform validation in smaller chunks if memory is an issue
                        with torch.no_grad():
                            for model_input, gt in val_dataloader:
                                # Move data to the specified device
                                model_input, gt = model_input.to(device), gt.to(device)

                                # Forward pass
                                try:
                                    model_output = model(model_input)
                                    val_loss = loss_fn(model_output, gt)
                                except RuntimeError as e:
                                    if "CUDA out of memory" in str(e):
                                        print("CUDA out of memory. Reducing batch size...")
                                        torch.cuda.empty_cache()
                                        continue  # Skip this batch
                                    else:
                                        raise e

                                # Store loss
                                val_losses.append(val_loss.item())

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)