import torch
from logging import getLogger
from utils import compute_average_distance_for_knn

logger = getLogger()

def get_translation_candidates(source_embeddings, target_embeddings, params):
    """
    Retrieve the best translation pair candidates between source and target embeddings.
    """
    batch_size = 128

    all_similarity_scores = []
    all_best_targets = []

    # Number of source words to consider
    num_source_words = source_embeddings.size(0)
    if params.dico_max_rank > 0 and not params.dico_method.startswith('invsm_beta_'):
        num_source_words = min(params.dico_max_rank, num_source_words)

    # Nearest neighbors method
    if params.dico_method == 'nn':
        for batch_start_idx in range(0, num_source_words, batch_size):
            # Compute similarity scores between the source words in the batch and all target words
            similarity_scores = target_embeddings.mm(source_embeddings[batch_start_idx:min(num_source_words, batch_start_idx + batch_size)].transpose(0, 1)).transpose(0, 1)
            top_scores, top_target_indices = similarity_scores.topk(2, dim=1, largest=True, sorted=True)

            # Append scores and target indices for the current batch
            all_similarity_scores.append(top_scores.cpu())
            all_best_targets.append(top_target_indices.cpu())

        all_similarity_scores = torch.cat(all_similarity_scores, 0)
        all_best_targets = torch.cat(all_best_targets, 0)

    # Inverted softmax method
    elif params.dico_method.startswith('invsm_beta_'):
        beta = float(params.dico_method[len('invsm_beta_'):])

        for batch_start_idx in range(0, target_embeddings.size(0), batch_size):
            similarity_scores = source_embeddings.mm(target_embeddings[batch_start_idx:batch_start_idx + batch_size].transpose(0, 1))
            similarity_scores.mul_(beta).exp_()
            similarity_scores.div_(similarity_scores.sum(0, keepdim=True).expand_as(similarity_scores))

            top_scores, top_target_indices = similarity_scores.topk(2, dim=1, largest=True, sorted=True)

            # Update scores and targets
            all_similarity_scores.append(top_scores.cpu())
            all_best_targets.append((top_target_indices + batch_start_idx).cpu())

        all_similarity_scores = torch.cat(all_similarity_scores, 1)
        all_best_targets = torch.cat(all_best_targets, 1)

        all_similarity_scores, top_target_indices = all_similarity_scores.topk(2, dim=1, largest=True, sorted=True)
        all_best_targets = all_best_targets.gather(1, top_target_indices)

    # CSLS (Cross-domain Similarity Local Scaling) method
    elif params.dico_method.startswith('csls_knn_'):
        knn = int(params.dico_method[len('csls_knn_'):])

        avg_distance_source = torch.from_numpy(compute_average_distance_for_knn(target_embeddings, source_embeddings, knn))
        avg_distance_target = torch.from_numpy(compute_average_distance_for_knn(source_embeddings, target_embeddings, knn))

        print(f"Shape {avg_distance_source.shape}")
        print(f"Shape {avg_distance_target.shape}")
        avg_distance_source = avg_distance_source.type_as(source_embeddings)
        avg_distance_target = avg_distance_target.type_as(target_embeddings)

        for batch_start_idx in range(0, num_source_words, batch_size):
            similarity_scores = target_embeddings.mm(source_embeddings[batch_start_idx:min(num_source_words, batch_start_idx + batch_size)].transpose(0, 1)).transpose(0, 1)
            print(f"Shape {similarity_scores.shape}")
            similarity_scores.mul_(2)
            print(f"Shape {avg_distance_source[batch_start_idx:min(num_source_words, batch_start_idx+batch_size)][:, None].shape}")
            print(f"Shape {avg_distance_target[None, :].shape}")
            similarity_scores.sub_(avg_distance_source[batch_start_idx:min(num_source_words, batch_start_idx + batch_size)][:, None] + avg_distance_target[None, :])
            
            top_scores, top_target_indices = similarity_scores.topk(2, dim=1, largest=True, sorted=True)

            all_similarity_scores.append(top_scores.cpu())
            all_best_targets.append(top_target_indices.cpu())

        all_similarity_scores = torch.cat(all_similarity_scores, 0)
        all_best_targets = torch.cat(all_best_targets, 0)

    translation_pairs = torch.cat([
        torch.arange(0, all_best_targets.size(0)).long().unsqueeze(1),
        all_best_targets[:, 0].unsqueeze(1)
    ], 1)

    # Sanity check
    assert all_similarity_scores.size() == translation_pairs.size() == (num_source_words, 2)

    # Sort pairs by score confidence
    confidence_diff = all_similarity_scores[:, 0] - all_similarity_scores[:, 1]
    sorted_indices = confidence_diff.sort(0, descending=True)[1]
    all_similarity_scores = all_similarity_scores[sorted_indices]
    translation_pairs = translation_pairs[sorted_indices]

    # Apply maximum rank filter
    if params.dico_max_rank > 0:
        rank_mask = translation_pairs.max(1)[0] <= params.dico_max_rank
        mask = rank_mask.unsqueeze(1).expand_as(all_similarity_scores).clone()
        all_similarity_scores = all_similarity_scores.masked_select(mask).view(-1, 2)
        translation_pairs = translation_pairs.masked_select(mask).view(-1, 2)

    # Apply maximum dictionary size
    if params.dico_max_size > 0:
        all_similarity_scores = all_similarity_scores[:params.dico_max_size]
        translation_pairs = translation_pairs[:params.dico_max_size]

    # Apply minimum dictionary size
    if params.dico_min_size > 0:
        confidence_diff[:params.dico_min_size] = 1e9

    # Apply confidence threshold
    if params.dico_threshold > 0:
        threshold_mask = confidence_diff > params.dico_threshold
        logger.info(f"Selected {threshold_mask.sum()} / {confidence_diff.size(0)} pairs above the confidence threshold.")
        threshold_mask = threshold_mask.unsqueeze(1).expand_as(translation_pairs).clone()
        translation_pairs = translation_pairs.masked_select(threshold_mask).view(-1, 2)

    return translation_pairs


def build_translation_dictionary(src_emb, tgt_emb, params, s2t_candidates=None, t2s_candidates=None):
    """
    Build a training dictionary using source and target embeddings.
    """
    logger.info("Building the training dictionary ...")
    source_to_target = 'S2T' in params.dico_build
    target_to_source = 'T2S' in params.dico_build
    assert source_to_target or target_to_source

    if source_to_target:
        if s2t_candidates is None:
            s2t_candidates = get_translation_candidates(src_emb, tgt_emb, params)
    if target_to_source:
        if t2s_candidates is None:
            t2s_candidates = get_translation_candidates(tgt_emb, src_emb, params)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if params.dico_build == 'S2T':
        final_dict = s2t_candidates
    elif params.dico_build == 'T2S':
        final_dict = t2s_candidates
    else:
        s2t_set = set([(a, b) for a, b in s2t_candidates.numpy()])
        t2s_set = set([(a, b) for a, b in t2s_candidates.numpy()])
        if params.dico_build == 'S2T|T2S':
            final_pairs = s2t_set | t2s_set
        else:
            assert params.dico_build == 'S2T&T2S'
            final_pairs = s2t_set & t2s_set
            if len(final_pairs) == 0:
                logger.warning("Empty intersection ...")
                return None
        final_dict = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    logger.info(f'New training dictionary contains {final_dict.size(0)} pairs.')
    return final_dict.to(params.device) if params.device else final_dict
