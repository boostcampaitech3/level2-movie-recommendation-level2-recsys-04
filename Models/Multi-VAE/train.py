import pickle

from torch import optim

from utils import *
from metrics import *
from dataset import DataLoader
from models import *


def make_data(raw_data, args):
    raw_data, user_activity, item_activity = filter_triplets(raw_data, 5, 0)

    # Shuffle User Indices
    unique_uid = user_activity.index
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    n_heldout_users = args.heldout
    # Split train/valid
    train_users = unique_uid[:(n_users - n_heldout_users)]
    valid_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['user'].isin(train_users)]
    unique_iid = train_plays['item'].unique()

    item2id = dict((iid, i) for i, iid in enumerate(unique_iid))
    user2id = dict((uid, i) for i, uid in enumerate(unique_uid))

    pro_dir = os.path.join(args.data_dir, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'item2id.pkl'), 'wb') as f:
        pickle.dump(item2id, f)

    with open(os.path.join(pro_dir, 'user2id.pkl'), 'wb') as f:
        pickle.dump(user2id, f)

    with open(os.path.join(pro_dir, 'unique_iid.txt'), 'w') as f:
        for iid in unique_iid:
            f.write(f"{iid}\n")

    val_plays = raw_data.loc[raw_data['user'].isin(valid_users)]
    val_plays = val_plays.loc[val_plays['item'].isin(unique_iid)]
    val_plays_tr, val_plays_te = split_train_test_proportion(val_plays, args.val_ratio)

    train_data = numerize(train_plays, user2id, item2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    val_data_tr = numerize(val_plays_tr, user2id, item2id)
    val_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    val_data_te = numerize(val_plays_te, user2id, item2id)
    val_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    print("Make train valid data done!")


def main(args):
    # Set random seed
    seed_everything(args.seed)

    raw_data = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'), header=0)
    print("Make train valid data")
    make_data(raw_data, args)

    # load data
    loader = DataLoader(args.data_dir)

    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    val_data_tr, val_data_te = loader.load_data('validation')

    N = train_data.shape[0]
    idx_list = list(range(N))
    e_N = val_data_tr.shape[0]
    e_idx_list = list(range(e_N))

    # Build model
    p_dims = [200, 600, n_items]
    model = MultiVAE(p_dims).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = loss_function_vae

    # Training and Valid
    best_n100 = -np.inf
    best_r10 = -np.inf
    update_count = 0
    print("Training Start!")
    print("-" * 89)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        np.random.shuffle(idx_list)

        for idx, start_idx in enumerate(range(0, N, args.batch_size)):
            end_idx = min(start_idx + args.batch_size, N)
            data = train_data[idx_list[start_idx:end_idx]]
            data = naive_sparse2tensor(data).to(args.device)
            optimizer.zero_grad()

            if args.step_size > 0:
                anneal = min(args.anneal_cap, 1. * update_count/args.step_size)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data)
            loss = criterion(recon_batch, data, mu, logvar, anneal)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            update_count += 1

            if idx % args.log_interval == 0 and idx > 0:
                print(f"| epoch {epoch:3} | {idx:4d}/{N//args.batch_size:4d} batches | "
                      f"loss {train_loss/args.log_interval:4.2f}")
                train_loss = 0.0

        # validation part
        model.eval()
        val_loss = 0.0
        n100_list = []
        r10_list = []
        r20_list = []
        r50_list = []

        with torch.no_grad():
            for start_idx in range(0, e_N, args.batch_size):
                end_idx = min(start_idx + args.batch_size, N)
                data = val_data_tr[e_idx_list[start_idx:end_idx]]
                held_out_data = val_data_te[e_idx_list[start_idx:end_idx]]
                data_tensor = naive_sparse2tensor(data).to(args.device)

                if args.step_size > 0:
                    anneal = min(args.anneal_cap, 1. * update_count / args.step_size)
                else:
                    anneal = args.anneal_cap

                recon_batch, mu, logvar = model(data_tensor)
                loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

                val_loss += loss.item()

                # Exclude examples from training set
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[data.nonzero()] = -np.inf

                n100 = NDCG_binary_at_k_batch(recon_batch, held_out_data, 100)
                r10 = Recall_at_k_batch(recon_batch, held_out_data, 10)
                r20 = Recall_at_k_batch(recon_batch, held_out_data, 20)
                r50 = Recall_at_k_batch(recon_batch, held_out_data, 50)

                n100_list.append(n100)
                r10_list.append(r10)
                r20_list.append(r20)
                r50_list.append(r50)

        val_loss /= (e_N // args.batch_size)
        n100 = np.mean(np.concatenate(n100_list))
        r10 = np.mean(np.concatenate(r10_list))
        r20 = np.mean(np.concatenate(r20_list))
        r50 = np.mean(np.concatenate(r50_list))

        print_str = f"| end of epoch {epoch:3d} | valid loss {val_loss:4.2f} | n100 {n100:5.3f} | " \
                    f"r10 {r10:5.3f} | r20 {r20:5.3f} | r50 {r50:5.3f}"

        if r10 > best_r10:
            with open(os.path.join(args.model_dir, args.save), 'wb') as f:
                torch.save(model, f)
            best_r10 = r10
            best_n100 = n100
            print_str += " ---> best model save!"

        print("-" * 89)
        print(print_str)
        print("-" * 89)


if __name__ == '__main__':
    args = args_getter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    print(f"Use device : {device}")

    main(args)