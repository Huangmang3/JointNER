# -*- coding: utf-8 -*-

from collections import Counter


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class AttachmentMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.eps = eps

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        lens = mask.sum(1)
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


class SpanMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.nertp = 0.0
        self.propntp = 0.0
        self.postp = 0.0
        self.goldtp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.nerpred = 0.0
        self.nergold = 0.0
        self.propnpred = 0.0
        self.propngold = 0.0
        self.pospred = 0.0
        self.posgold = 0.0
        self.eps = eps
        self.not_top_error = 0.0

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            # for span in pred:
            #     if span[0]==0 and span[1]==2:
            #         print(span)
            #     if span[0]==12 and span[1]==16:
            #         print(span)
            # ner span
            nerpred, nergold = Counter([tuple((span[0],span[1],span[2][:span[2].index("-")])) for span in pred if '-' in span[-1]]), Counter([tuple((span[0],span[1],span[2][:span[2].index("-")])) for span in gold if '-' in span[-1]])
            propngold =Counter([tuple((span[0],span[1],'PROPN')) for span in gold if '-' in span[-1]])
            # propnpred =Counter([tuple((span[0],span[1],'PROPN')) for span in pred if span[-1] in ['LOC-PROPN', 'PER-PROPN', 'ORG-PROPN','GPE-PROPN']])
            allpropn= Counter([tuple((span[0],span[1],'PROPN')) for span in pred if span[-1] == 'PROPN'])
            tmpj=-1
            top_propn_lst=[]
            for span in pred:
                if span[0]>=tmpj:
                    tmpj=span[1]
                    if 'PROPN' in span[-1]:
                        top_propn_lst.append(tuple((span[0],span[1],'PROPN')))
            propnpred=Counter(top_propn_lst)
            for span1 in propngold:
                if span1 in allpropn and span1 not in propnpred:
                    self.not_top_error+=1
            # pos span
            pospred, posgold = Counter([tuple((span[0],span[1],span[2][span[2].index("-")+1:]) if "-" in span[2] else span) for span in pred]), Counter([tuple((span[0],span[1],span[2][span[2].index("-")+1:]) if "-" in span[2] else span) for span in gold])
            # multi-grained word span
            upred, ugold = Counter([tuple(span[:-1]) for span in pred]), Counter([tuple(span[:-1]) for span in gold if span[-1]=="W"])
            # total labeled span
            lpred, lgold = Counter([tuple(span) for span in pred]), Counter([tuple(span) for span in gold])
            utp, ltp = list((upred & ugold).elements()), list((lpred & lgold).elements()) # correct multi-grained word, correct labeled span
            nertp = list((nerpred & nergold).elements()) # correct ne
            propntp = list((propnpred & propngold).elements())
            postp = list((pospred & posgold).elements()) # correct pos
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.nertp += len(nertp)
            self.propntp += len(propntp)
            self.postp += len(postp)
            self.pred += len(pred)
            self.gold += len(gold)
            self.nerpred += len(nerpred)
            self.nergold += len(nergold)
            self.propnpred += len(propnpred)
            self.propngold += len(propngold)
            self.pospred += len(pospred)
            self.posgold = self.posgold + len(posgold) - len(nerpred)
            # self.posgold += len(posgold)
        return self

    def __repr__(self):
        s = f"NER_P: {self.nerp:6.2%} NER_R: {self.nerr:6.2%} NER_F: {self.nerf:6.2%} "
        s += f"PROPN_P: {self.pp:6.2%} PROPN_R: {self.pr:6.2%} PROPN_F: {self.pf:6.2%} "
        s += f"POS_P: {self.posp:6.2%} POS_R: {self.posr:6.2%} POS_F: {self.posf:6.2%} "
        s += f"MWS_P: {self.up:6.2%} MWS_R: {self.ur:6.2%} MWS_F: {self.uf:6.2%} "
        s += f"TOTAL_P: {self.lp:6.2%} TOTAL_R: {self.lr:6.2%} TOTAL_F: {self.lf:6.2%} "
        s += f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"NERTP: {self.nertp/100:6.2%} NERPRED: {self.nerpred/100:6.2%} PROPNPRED: {self.propnpred/100:6.2%} NERGOLD: {self.nergold/100:6.2%} NOT_TOP_ERROR: {self.not_top_error/100:6.2%}"
        return s

    @property
    def score(self):
        # return self.lf
        return self.nerf
        # return self.uf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)

    @property
    def nerp(self):
        return self.nertp / (self.nerpred + self.eps)

    @property
    def nerr(self):
        return self.nertp / (self.nergold + self.eps)

    @property
    def nerf(self):
        return 2 * self.nertp / (self.nerpred + self.nergold + self.eps)

    @property
    def pp(self):
        return self.propntp / (self.propnpred + self.eps)

    @property
    def pr(self):
        return self.propntp / (self.propngold + self.eps)

    @property
    def pf(self):
        return 2 * self.propntp / (self.propnpred + self.propngold + self.eps)

    @property
    def posp(self):
        return self.postp / (self.pospred + self.eps)

    @property
    def posr(self):
        return self.postp / (self.posgold + self.eps)

    @property
    def posf(self):
        return 2 * self.postp / (self.pospred + self.posgold + self.eps)


class ChartMetric(Metric):

    def __init__(self, eps=1e-12):
        super(ChartMetric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()
        return self

    def __repr__(self):
        return f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)
