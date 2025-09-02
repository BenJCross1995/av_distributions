import heapq
import math
import numpy as np
from collections import defaultdict, deque
from itertools import chain, islice


def ngrams(iterable, n):
    """
    Yields all n-grams of a sequence
    """
    it = iter(iterable)
    window = deque(islice(it, n-1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)


def evergrams(iterable, n):
    """
    Yields all 1-, 2-, ..., n-grams of a sequence
    """
    window = deque(maxlen=n)
    for x in iterable:
        window.append(x)
        t = tuple(window)
        for i in range(n):
            if i<len(t):
                yield t[i:]


class KneserNeyLanguageModel:
    def __init__(self, order, discount=0.5, pad_start_element="<s>", pad_end_element="</s>", unk_element="<unk>", pad_start_count=None, pad_end_count=1, special_handling_of_pad_start_element=False):
        """
        A language model with 'kneser-ney' smoothing.
        Args:
            order: the maximum size of ngrams that will be considered (a context size is order-1 or lower)
            discount: the discount value (between 0 and 1) of the kneser-ney smoothing
            pad_start_element: a special element (aka. BOS-token), that is added to the start of a sequence while training
            pad_end_element: a special element (aka. EOS-token), that is added to the end of a sequence while training
            unk_element: a special element (aka. OOV-token), that is used in inference when an element is seen, that has not been part of trainings sequences
            pad_start_count: the number of `pad_start_element` which are added to the start of a sequence while training (if None, order-1 is used)
            pad_start_count: the number of `pad_end_element` which are added to the end of a sequence while training
            special_handling_of_pad_start_element: if true, the pad_start_element is handled differently, by excluding it from the vocabulary and forcing its probability to be zero
        """
        assert order > 0
        assert 1 >= discount >= 0
        assert pad_start_count == None or pad_start_count >= 0
        assert pad_end_count >= 0 

        self.order = order
        self.discount = discount
        self.pad_start_element = pad_start_element
        self.pad_end_element = pad_end_element
        self.unk_element = unk_element
        self.pad_start_count = order-1 if pad_start_count is None else pad_start_count
        self.pad_end_count = pad_end_count
        self.special_handling_of_pad_start_element = special_handling_of_pad_start_element

        self.voc = set()
        self.voc_size = 0                   # this is V
        self.context_element_count = defaultdict(lambda: defaultdict(int)) # this is used for c(gw)
        self.gram_count = defaultdict(int)  # this is used for c(g)
        self.pre_voc = defaultdict(set)     # this is used for N1+(•g)
        self.suc_voc = defaultdict(set)     # this is used for N1+(g•)
        self.pre_suc_voc = defaultdict(set) # this is used for N1+(•g•)
        self.total_element_count = 0        # this is 𝚺_w c(w)


    def pad_sequence(self, sequence):
        """
        Pads a sequence according to the settings. 
        Args:
            sequence: the sequence to be padded
        """
        return chain(
            [self.pad_start_element] * self.pad_start_count,
            sequence,
            [self.pad_end_element] * self.pad_end_count
        )


    def fit(self, sequence, pad_sequence=True):
        """
        Trains/Fits the language model on a single sequence, which gets padded internally. Note, that the internal model is updated progressively (i.e. the model is not cleared beforehand).
        Args:
            sequence: the sequence to be fitted
            pad_sequence: if true, pad the given sequence
        """
        if pad_sequence:
            padded_sequence = self.pad_sequence(sequence)
            
        for gram in evergrams(padded_sequence, self.order):

            if self.special_handling_of_pad_start_element:
                if gram[-1] == self.pad_start_element: 
                    # This is a current "hack" for the "special handling if BOS" (i.e. in order to not include the BOS in the vocabulary)
                    continue

            self.gram_count[gram] += 1
            self.context_element_count[gram[:-1]][gram[-1]] += 1
            self.pre_voc[gram[1:]].add(gram[0])
            self.suc_voc[gram[:-1]].add(gram[-1])
            if len(gram) >= 2:
                self.pre_suc_voc[gram[1:-1]].add((gram[0], gram[-1]))
            if len(gram) == 1:
                self.voc.add(gram[0])
                self.total_element_count += 1
        self.voc.add(self.unk_element)
        self.voc_size = len(self.voc)
        
        if self.special_handling_of_pad_start_element:
            # Add +1 counts to <s>, <s><s>, etc. counters
            g = (self.pad_start_element,)
            for _ in range(self.order):
                self.gram_count[g] += 1
                g += (self.pad_start_element,)

    def fit_corpus(self, corpus, **kwargs):
        for seq in corpus:
            self.fit(seq, **kwargs)

    def probability(self, element, context=()):
        """
        Calculates and returns the probability of the given element to follow the given context according to the model.
        Args:
            element: a single element (such as a token)
            context: a sequence of elements (such as a token sequence)
        """
        if self.special_handling_of_pad_start_element and element == self.pad_start_element:
            return 0.0
        if not isinstance(context, tuple):
            context = tuple(context) 
        if len(context) >= self.order:
            context = context[-self.order+1:] # trim the context if it is longer than order-1. Is this correct?
        
        def prob_star(element, context):
            if len(context) >= 1 and self.gram_count[context] == 0:
                return prob_star(element, context[1:])
            
            p_disc = max(0, len(self.pre_voc[context + (element,)]) - self.discount) / len(self.pre_suc_voc[context])
            interp = (self.discount * len(self.suc_voc[context]) ) / len(self.pre_suc_voc[context])
            
            if len(context) == 0:
                p_interp = 1. / self.voc_size
            else:
                p_interp = prob_star(element, context[1:])
            
            return p_disc + interp * p_interp
        
        if len(context) >= 1 and self.gram_count[context] == 0:
            return prob_star(element, context[1:])
        elif len(context) == 0:
            return (
                max(0, self.gram_count[(element,)] - self.discount) / self.total_element_count + 
                (self.discount * len(self.suc_voc[()])) / self.total_element_count / self.voc_size
            )
        else:
            return (
                max(0, self.gram_count[context + (element,)] - self.discount) / self.gram_count[context] + 
                (self.discount * len(self.suc_voc[context]) ) / self.gram_count[context] * prob_star(element, context[1:])
            )
        
    
    def probabilities(self, sequence, pad_sequence=True):
        """
        Calculates and returns the probabilities of the elements in the given sequence according to the model.
        Args:
            sequence: a sequence of elements (such as a token sequence) of which the perplexity is calculated
            pad_sequence: if true, pad the given sequence
        """
        if pad_sequence:
            sequence = self.pad_sequence(sequence)
        return [self.probability(gram[-1], gram[:-1]) for gram in ngrams(sequence, self.order)]
    
    
    def perplexity(self, sequence, pad_sequence=True):
        """
        Calculates and returns the perplexity of the given sequence according to the model.
        Args:
            sequence: a sequence of elements (such as a token sequence) of which the perplexity is calculated
            pad_sequence: if true, pad the given sequence
        """
        probs = self.probabilities(sequence, pad_sequence=pad_sequence)
        log_probs = [math.log2(prob) for prob in probs]
        return 2**(-np.mean(log_probs))


    def generate(self, max_element_count, start_sequence=None, return_pad=False):
        """
        Generates a sequence or continues a given start sequence according to the model.
        Args:
            max_element_count: the maximum number of elements to be generated
            start_sequence: a sequence from which the generation process is started (if None, a sequence of multiple pad_start_element is used)
            return_pad: if true, the returned sequence will include pad_start_element and pad_end_element
        """
        sequence = (self.pad_start_element,) * self.pad_start_count if start_sequence is None else tuple(start_sequence)
        for _ in range(max_element_count):
            voc = list(self.voc)
            voc_probas = [self.probability(v, sequence) for v in voc]
            next_element = np.random.choice(voc, p=voc_probas)
            sequence = sequence + (next_element,)
            if next_element == self.pad_end_element:
                break
        if return_pad:
            return sequence
        else: 
            return tuple(t for t in sequence if t not in [self.pad_start_element, self.pad_end_element])

    def _top_n_next(self, context, n):
        """
        Internal: get the top-n tokens (and their probs) after `context`.
        """
        # normalize context exactly like .probability()
        if not isinstance(context, tuple):
            context = tuple(context)
        if len(context) >= self.order:
            context = context[-(self.order-1):]

        # score every token
        scores = ((tok, self.probability(tok, context)) for tok in self.voc)
        # pick top-n
        return heapq.nlargest(n, scores, key=lambda x: x[1])

    def probabilities_top_n(self, sequence, n=10, pad_sequence=True):
        """
        Like .probabilities(), but at each n-gram returns the top-n (token, prob).
        Returns a list of the same length as .probabilities(): one list of n tuples per position.
        """
        if pad_sequence:
            seq = list(self.pad_sequence(sequence))
        else:
            seq = list(sequence)

        out = []
        for gram in ngrams(seq, self.order):
            ctx = gram[:-1]
            out.append(self._top_n_next(ctx, n))
        return out

if __name__ == "__main__": #Example usage

    lm = KneserNeyLanguageModel(order=4, discount=0.5, special_handling_of_pad_start_element=False)
    lm.fit("a a b a a b a b a b a b".split())

    print("P(a)     =", lm.probability("a"))
    print("P(a|b)   =", lm.probability("a", ("b",)))
    print("P(b|a)   =", lm.probability("b", ("a",)))
    print("P(b|a a) =", lm.probability("b", ("a", "a",)))
    print("P(c)     =", lm.probability("c"))
    print("P(c|a)   =", lm.probability("c", ("a",)))
    
    print(f"{lm.perplexity('a a b a a b a b a b a b c'.split())=}")

    print("Generated sequence:", lm.generate(100))
