import torch
import argparse
from tqdm import tqdm
from functools import partial
from torch.optim import AdamW
from torch.utils.data import DataLoader
from common.dataset import MultiNDataset
from torch.nn.utils import clip_grad_norm_
from common.evaluation_utils import get_validation_mAP
from transformers import get_linear_schedule_with_warmup
from common.model_utils import  get_model, average_pool, calculate_mnrl_loss, JobBERTEncoder
from common.preprocessing_utils import get_formatter, get_train_data, get_templates, custom_collate_fn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--warmup', type=float, default=0.06, help='Warmup ratio for the learning rate scheduler')
    parser.add_argument('--model_name', type=str, default='e5_instruct', help='Choose a pretrained enocder model to finetune: escoxlm_r, e5_large, e5_instruct', choices=['escoxlm_r', 'e5_large', 'e5_instruct'])
    parser.add_argument('--augmentation', type=str, default='llm', help='Choose a description augmentation: esco, llm, no_desc', choices=['esco', 'llm', 'no_desc'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint to resume training from')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    betas = (0.9, 0.98)
    gradient_accumulation_steps = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    train_data = get_train_data()
    print(f"Loaded {len(train_data)} training pairs.")
    print(f"Loading model: {args.model_name}")
    encoder, tokenizer = get_model(args.model_name, device)
    print(f"Model {args.model_name} loaded successfully.")
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        encoder.load_state_dict(checkpoint['weights'])
        best_val_mAP = checkpoint['mAP']
        print(f"Loaded checkpoint with mAP: {best_val_mAP}")
    else:
        best_val_mAP = 0
    job_template, skill_template = get_templates(args.model_name, True if args.augmentation in ['esco', 'llm'] else False)
    print(f"Using templates: job_template='{job_template}', skill_template='{skill_template}'")
    formatter = get_formatter(args.augmentation)
    formatter.initialize(job_template, skill_template)
    train_dataset = MultiNDataset(train_data, formatter)
    collate_wrapper = partial(custom_collate_fn, tokenizer=tokenizer, max_length=256)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper)

    if args.augmentation == 'esco':
        print('Downloading JobBERT weights...')
        JobBERTEncoder.initialize(device=device)

    optim = AdamW(encoder.parameters(), lr=args.lr, betas=betas)
    num_training_steps = (len(train_loader) * args.epochs) // gradient_accumulation_steps
    num_warmup_steps = int(args.warmup * num_training_steps)  # 6% warmup
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    global_step = 0
    validation_mAP = 0
    best_val_mAP = 0

    encoder.zero_grad()
    print("Starting training...")
    print(f"Number of training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}, Model name: {args.model_name}, Augmentation: {args.augmentation}")
    for epoch in range(args.epochs):
        encoder.train()
        formatter.set_context('train')
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", postfix={'last mAP': validation_mAP, 'best mAP': best_val_mAP}) # Keep pbar after loop
        for step, batch in enumerate(pbar):
            job_inputs = {k: v.to(device) for k, v in batch['jobs'].items()}
            skill_inputs = {k: v.to(device) for k, v in batch['skills'].items()}

            job_outputs = encoder(**job_inputs)
            skill_outputs = encoder(**skill_inputs)

            job_embeddings = average_pool(job_outputs.last_hidden_state, job_inputs['attention_mask'])
            skill_embeddings = average_pool(skill_outputs.last_hidden_state, skill_inputs['attention_mask'])

            loss = calculate_mnrl_loss(job_embeddings, skill_embeddings, temperature=0.05)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad() 
                global_step += 1

                avg_loss_so_far = train_loss / (step + 1)
                pbar.set_postfix({'loss': f"{avg_loss_so_far:.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}", 'last mAP': validation_mAP, 'best mAP': best_val_mAP})

        avg_epoch_loss = train_loss / len(train_loader)
        
        encoder.eval()
        formatter.set_context('val')
        validation_mAP = get_validation_mAP(encoder, tokenizer, formatter, device)
        
        print(f'Epoch {epoch+1} finished. Train loss: {avg_epoch_loss}, Validation mAP: {validation_mAP}')
        if validation_mAP>best_val_mAP:
            torch.save({'mAP': validation_mAP, 'weights': encoder.state_dict()}, f'weights/best_model_{args.model_name}_{args.augmentation}_mAP.pt')
            best_val_mAP = validation_mAP