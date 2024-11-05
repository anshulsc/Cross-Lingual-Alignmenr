import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm  # Import tqdm for progress bars

from utils import bool_flag, initialize_exp
from model import build_model
from trainer import Trainer
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="muse/", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="final", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='hi', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=100000, help="Maximum vocabulary size (-1 to disable)")
# mappiconng
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0.4, help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=10000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.05, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=100000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_train", type=str, default="identical_char", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=20000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--lr", type=int, default=3e-4, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--weight_decay", type=int, default=1e-4, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--gradient_clip_val", type=int, default=1.0, help="Maximum generated dictionary size (0 to disable)")

# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="/Users/anshulsingh/lockedin/cross-lingual-alignment/embeddings/pretrained/cc.en.300.bin", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="/Users/anshulsingh/lockedin/cross-lingual-alignment/embeddings/pretrained/cc.hi.300.bin", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters

assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build model / trainer / evaluator
logger = initialize_exp(params)
params.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)
trainer.load_training_dictionary(params.dico_train)


"""
Learning loop for Adversarial Training
"""
if params.adversarial:
    logger.info('----> ADVERSARIAL TRAINING <----\n\n')

    # training loop with tqdm progress bar
    for n_epoch in range(params.n_epochs):
        logger.info(f'Starting adversarial training epoch {n_epoch}...')
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_COSTS': []}

        with tqdm(total=params.epoch_size, desc=f"Epoch {n_epoch + 1}/{params.n_epochs}", unit='batch') as pbar:
            for n_iter in range(0, params.epoch_size, params.batch_size):

                # discriminator training
                for _ in range(params.dis_steps):
                    trainer.train_discriminator(stats)

                # mapping training (discriminator fooling)
                n_words_proc += trainer.train_mapping(stats)

                # Update progress bar and log stats every 500 iterations
                if n_iter % 500 == 0:
                    stats_str = [('DIS_COSTS', 'Discriminator loss')]
                    stats_log = ['%s: %.4f' % (v, np.mean(stats[k])) for k, v in stats_str if len(stats[k]) > 0]
                    stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                    logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

                    # reset stats
                    tic = time.time()
                    n_words_proc = 0
                    for k, _ in stats_str:
                        del stats[k][:]

                # Update progress bar
                pbar.update(params.batch_size)

        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best_model()
        logger.info(f'End of epoch {n_epoch}.\n\n')

        # Update learning rate (stop if it's too small)
        if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break


"""
Learning loop for Procrustes Iterative Refinement
"""
if params.n_refinement > 0:
    logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
    trainer.reload_best_model()

    # refinement loop with tqdm progress bar
    for n_iter in range(params.n_refinement):
        logger.info(f'Starting refinement iteration {n_iter}...')

        # build dictionary from aligned embeddings
        trainer.build_translation_dictionary()

        # apply Procrustes solution
        trainer.solve_procrustes()

        # log refinement progress
        to_log = OrderedDict({'n_iter': n_iter})

        # JSON log / save best model / end of iteration
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best_model()
        logger.info(f'End of refinement iteration {n_iter}.\n\n')


# export embeddings
if params.export:
    trainer.reload_best_model()
    trainer.export_embeddings()
