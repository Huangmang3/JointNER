
from parser.utils.fn import stripe
import torch
import torch.autograd as autograd

@torch.enable_grad()
def crf(scores, mask, target=None, mbr=False):
    """[summary]

    Args:
        scores (Tensor(B, seq_len, seq_len, n_labels))
        mask (Tensor(B, seq_len, seq_len))
        target (Tensor(B, seq_len, seq_len)): Defaults to None.
        marg (bool, optional): Defaults to False.

    Returns:
        crf-loss, marginal probability for spans
    """
    # (B)
    lens = mask[:, 0].sum(-1)
    total = lens.sum()
    batch_size, seq_len, seq_len, n_labels = scores.shape
    # in eval(), it's false; and in train(), it's true
    training = scores.requires_grad

    s = inside_simple(scores.requires_grad_(), mask)
    logZ = s[0].gather(0, lens.unsqueeze(0)).sum()
    
    probs = scores
    if mbr:
        probs, = autograd.grad(logZ, scores, retain_graph=training)
    if target is None:
        return probs
    # TODO target -> (B, seq_len, seq_len, 3)
    
    span_mask = target.ge(0) & mask
    total = span_mask.sum()
    # (T, n_labels)
    scores = scores[span_mask] 
    # (T, 1)
    target = target[span_mask].unsqueeze(-1)
    # TODO why / total?
    # TODO int8 for index
    loss = (logZ - scores.gather(1, target).sum()) / total
    return loss, probs

def inside_simple(scores, mask):
    """Simple inside algorithm as supar.

    Args:
        scores (Tensor(B, seq_len, seq_len, n_labels))
        trans_mask (Tensor(n_labels, n_labels, n_labels)): boolen value
            (i, j, k) == 0 indicates k->ij is impossible
            (i, j, k) == 1 indicates k->ij is possible
        mask (Tensor(B, seq_len, seq_len))

    Returns:
        Tensor: [seq_len, seq_len, n_labels, batch_size]
    """
    # [batch_size, seq_len, seq_len]
    scores = scores.logsumexp(-1)
    batch_size, seq_len, seq_len = scores.shape
    # permute is convenient for diagonal which acts on dim1=0 and dim2=1
    # [seq_len, seq_len, batch_size]
    scores, mask = scores.permute(1, 2, 0), mask.permute(1, 2, 0)
    # s[i, j]: sub-tree spanning from i to j
    # [seq_len, seq_len, batch_size]
    s = torch.full_like(scores, float('-inf'))

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # diag_mask is used for ignoring the excess of each sentence
        # [batch_size, n]
        # diag_mask = mask.diagonal(w)

        if w == 1:
            # scores.diagonal(w): [n_labels, batch_size, n]
            # scores.diagonal(w).permute(1, 2, 0)[diag_mask]: (T, n_labels)
            # s.diagonal(w).permute(1, 2, 0)[diag_mask] = scores.diagonal(w).permute(1, 2, 0)[diag_mask]
            # no need  diag_mask
            # [n_labels, batch_size]
            s.diagonal(w).copy_(scores.diagonal(w))
            continue 
        
        # scores for sub-tree spanning from `i to k` and `k+1 to j`, considering all labels
        # NOTE: stripe considering all split points and spans with same width
        # stripe: [n, w-1, batch_size] 
        s_span = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
        # [batch_size, n, w-1]
        s_span = s_span.permute(2, 0, 1)
        if s_span.requires_grad:
            s_span.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
        # [batch_size, n]
        s_span = s_span.logsumexp(-1)
        # [batch_size, n] = [batch_size, n] +  [batch_size, n]
        s.diagonal(w).copy_(s_span + scores.diagonal(w))

    # [seq_len, seq_len, batch_size]
    return s

def cky_simple(scores, mask):
    """
    We can use max labels score as span's score,
    then use the same cky as two-stage.

    When backtracking, we get label as well.

    Args:
        scores (Tensor(B, seq_len, seq_len, n_labels))
        mask (Tensor(B, seq_len, seq_len))

    Returns:
        [[(i, j, l), ...], ...]
    """
    lens = mask[:, 0].sum(-1)
    # (B, seq_len, seq_len)
    scores, labels = scores.max(-1)
    # [seq_len, seq_len, batch_size]
    scores = scores.permute(1, 2, 0)
    seq_len, seq_len, batch_size = scores.shape
    s = scores.new_zeros(seq_len, seq_len, batch_size)
    p = scores.new_zeros(seq_len, seq_len, batch_size).long()

    for w in range(1, seq_len):
        n = seq_len - w
        # (1, n)
        starts = p.new_tensor(range(n)).unsqueeze(0)

        if w == 1:
            # scores.diagonal(w): [batch_size, n]
            s.diagonal(w).copy_(scores.diagonal(w))
            continue

        # [n, w-1, batch_size] 
        s_span = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
        # [batch_size, n, w]
        s_span = s_span.permute(2, 0, 1)
        # [batch_size, n]
        s_span, p_span = s_span.max(-1)
        s.diagonal(w).copy_(s_span + scores.diagonal(w))
        p.diagonal(w).copy_(p_span + starts + 1)

    def backtrack(p, i, j, labels):
        """span(i, j, l)

        Args:
            p (List[List]): backtrack points.
            labels (List[List]: [description]

        Returns:
            [type]: [description]
        """
        if j == i + 1:
            return [(i, j, labels[i][j])]
        split = p[i][j]
        ltree = backtrack(p, i, split, labels)
        rtree = backtrack(p, split, j, labels)
        # top-down, [(0, 9), (0, 6), (0, 3), ]
        return [(i, j, labels[i][j])] + ltree + rtree

    p = p.permute(2, 0, 1).tolist()
    labels = labels.tolist()

    trees = [backtrack(p[i], 0, length, labels[i])
             for i, length in enumerate(lens.tolist())]

    return trees