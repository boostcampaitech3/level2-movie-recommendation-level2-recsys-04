from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim

from utils import *
from metrics import *
from model import RecVAE


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1

    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)

    if shuffle:
        idx_list = np.arange(total_samples)
        np.random.shuffle(idx_list)
        idx_list = idx_list[:samples_per_epoch]
    else:
        idx_list = np.arange(samples_per_epoch)

    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idx_list[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)


def evaluate(model, data_in, data_out, metrics, args, samples_perc_per_epoch=1, batch_size=500):
    metrics = deepcopy(metrics)
    model.eval()

    for m in metrics:
        m['score'] = []

    for batch in generate(batch_size=batch_size, device=args.device, data_in=data_in, data_out=data_out,
                          samples_perc_per_epoch=samples_perc_per_epoch):
        rating_in = batch.get_ratings_to_dev()
        rating_out = batch.get_ratings(is_out=True)

        rating_pred = model(rating_in, cal_loss=False).cpu().detach().numpy()

        if not (data_in is data_out):
            rating_pred[batch.get_ratings().nonzero()] = -np.inf

        for m in metrics:
            m['score'].append(m['metric'](rating_pred, rating_out, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()

    return [metric['score'] for metric in metrics]


def run(model, opts, train_data, batch_size, epochs, beta, gamma, dropout_ratio, args):
    model.train()

    for epoch in range(1, epochs + 1):
        for batch in generate(batch_size=batch_size, device=args.device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()

            _, loss = model(ratings, beta=beta, gamma=gamma, dropout_ratio=dropout_ratio)
            loss.backward()

            for optimizer in opts:
                optimizer.step()


def main(args):
    # fixed random seed
    seed_everything(args.seed)

    data = get_data(args)

    # val_ratio == 0 means preprocessing heldout_user = 0
    if args.val_ratio == 0:
        train_data = data
    else:
        train_data, valid_in_data, valid_out_data = data

    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    input_dim = train_data.shape[1]

    model = RecVAE(hidden_dim, latent_dim, input_dim).to(args.device)
    model_best = RecVAE(hidden_dim, latent_dim, input_dim).to(args.device)

    learning_kwargs = {
        'model': model,
        'train_data': train_data,
        'batch_size': args.batch_size,
        'beta': args.beta,
        'gamma': args.gamma,
        'args': args
    }

    encoder_params = set(model.encoder.parameters())
    decoder_params = set(model.decoder.parameters())

    optimizer_encoder = optim.Adam(encoder_params, lr=args.lr)
    optimizer_decoder = optim.Adam(decoder_params, lr=args.lr)

    # train part
    # metrics = [{"metric": Recall_at_k_batch, "k": 10}]
    metrics = [{"metric": Recall_at_k_batch, "k": 10}]
    best_ndcg = -np.inf
    best_recall = -np.inf
    train_score, valid_score = [], []

    print("Training Start!")
    for epoch in range(1, args.epochs + 1):
        if args.not_alternating:
            run(opts=[optimizer_encoder, optimizer_decoder], epochs=1, dropout_ratio=0.5, **learning_kwargs)
        else:
            run(opts=[optimizer_encoder], epochs=args.enc_epochs, dropout_ratio=0.5, **learning_kwargs)
            model.update_prior()
            run(opts=[optimizer_decoder], epochs=args.dec_epochs, dropout_ratio=0, **learning_kwargs)

        train_score.append(evaluate(model=model, data_in=train_data, data_out=train_data, metrics=metrics, args=args,
                                    samples_perc_per_epoch=0.01)[0])

        if args.val_ratio != 0:
            valid_score.append(
                evaluate(model=model, data_in=valid_in_data, data_out=valid_out_data, metrics=metrics, args=args)[0]
            )

            if valid_score[-1] > best_recall:
                best_recall = valid_score[-1]

                with open(f'models/{args.save}_best.pt', 'wb') as f:
                    torch.save(model, f)

            print(f'epoch {epoch} | valid Recall@{metrics[0]["k"]}: {valid_score[-1]:.4f} | ' +
                  f'best valid: {best_ndcg:.4f} | train Recall@{metrics[0]["k"]}: {train_score[-1]:.4f}')

        print(f'epoch {epoch} | train Recall@{metrics[0]["k"]}: {train_score[-1]:.4f}')

    with open(f'models/{args.save}.pt', 'wb') as f:
        torch.save(model, f)


if __name__ == '__main__':
    args = args_getter()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Use device : {args.device}")

    main(args)
