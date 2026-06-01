#!/usr/bin/env python3
"""MobileBERT extractive question answering via TFLite.

Given a passage of text and a question, finds and returns the answer
span from within the passage.

Usage:
  python3 bert_qa.py --question "..." --context "..."
  python3 bert_qa.py --question "..." --file passage.txt
  python3 bert_qa.py  (interactive mode)
"""

import re
import sys
import time
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite

MODEL = '/home/glaeken/coral/mobilebert_qa.tflite'
VOCAB = '/home/glaeken/coral/bert_vocab.txt'
MAX_SEQ_LEN = 384
MAX_ANS_LEN = 32       # max tokens in a valid answer span
N_BEST      = 5        # candidates to consider when picking the answer

# ── tokenizer ────────────────────────────────────────────────────────────────

class WordpieceTokenizer:
    """Minimal BERT uncased WordPiece tokenizer."""

    def __init__(self, vocab_file):
        self.vocab = {}
        with open(vocab_file) as f:
            for i, line in enumerate(f):
                self.vocab[line.strip()] = i
        self.unk_id = self.vocab.get('[UNK]', 0)

    def tokenize(self, text):
        text = text.lower().strip()
        tokens = []
        for word in self._basic_tokenize(text):
            tokens.extend(self._wordpiece(word))
        return tokens

    def _basic_tokenize(self, text):
        # split on whitespace and punctuation
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        return text.split()

    def _wordpiece(self, word):
        if len(word) > 200:
            return ['[UNK]']
        if word in self.vocab:
            return [word]
        tokens, start = [], 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end] if start == 0 else '##' + word[start:end]
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                return ['[UNK]']
            tokens.append(found)
            start = end
        return tokens

    def convert(self, tokens):
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def inv_vocab(self):
        return {v: k for k, v in self.vocab.items()}

# ── encoding ─────────────────────────────────────────────────────────────────

def encode(tokenizer, question, context, max_seq_len=MAX_SEQ_LEN):
    """Build BERT QA input: [CLS] question [SEP] context [SEP], padded."""
    q_tokens = tokenizer.tokenize(question)
    c_tokens = tokenizer.tokenize(context)

    # reserve 3 slots for [CLS], [SEP], [SEP]
    max_ctx = max_seq_len - len(q_tokens) - 3
    c_tokens = c_tokens[:max_ctx]

    tokens   = ['[CLS]'] + q_tokens + ['[SEP]'] + c_tokens + ['[SEP]']
    seg_ids  = [0] * (len(q_tokens) + 2) + [1] * (len(c_tokens) + 1)
    ids      = tokenizer.convert(tokens)
    mask     = [1] * len(ids)

    # pad to max_seq_len
    pad = max_seq_len - len(ids)
    ids     += [0] * pad
    mask    += [0] * pad
    seg_ids += [0] * pad

    context_start = len(q_tokens) + 2   # index of first context token
    return (
        np.array([ids],     dtype=np.int32),
        np.array([mask],    dtype=np.int32),
        np.array([seg_ids], dtype=np.int32),
        tokens,
        context_start,
    )

# ── answer decoding ───────────────────────────────────────────────────────────

def decode_answer(tokens, start_logits, end_logits, context_start,
                  n_best=N_BEST, max_ans_len=MAX_ANS_LEN):
    """Return the best answer string and its score."""
    n = len(tokens)

    # collect n_best start + end positions within the context span
    top_starts = np.argsort(start_logits)[::-1]
    top_ends   = np.argsort(end_logits)[::-1]

    candidates = []
    for s in top_starts[:20]:
        for e in top_ends[:20]:
            if s < context_start or e < context_start:
                continue
            if e < s or e - s >= max_ans_len:
                continue
            candidates.append((s, e, start_logits[s] + end_logits[e]))

    if not candidates:
        return '(no answer found)', -1e9

    candidates.sort(key=lambda x: x[2], reverse=True)
    s, e, score = candidates[0]

    # convert tokens back to a readable string, merging ## continuations
    answer_tokens = tokens[s:e + 1]
    parts = []
    for t in answer_tokens:
        if t.startswith('##'):
            if parts:
                parts[-1] += t[2:]
            else:
                parts.append(t[2:])
        else:
            parts.append(t)
    answer = ' '.join(parts)
    # clean up spaces before punctuation
    answer = re.sub(r' ([.,!?;:\'\-])', r'\1', answer)
    return answer, score

# ── inference ─────────────────────────────────────────────────────────────────

def load_model(model_path):
    # MobileBERT is float32 — Edge TPU only accelerates quantized models,
    # so this always runs on CPU via tflite-runtime.
    interp = tflite.Interpreter(model_path)
    interp.allocate_tensors()
    return interp, 'CPU'

def ask(interp, tokenizer, question, context):
    ids, mask, seg_ids, tokens, ctx_start = encode(tokenizer, question, context)

    inp = {t['name']: t['index'] for t in interp.get_input_details()}
    interp.set_tensor(inp['input_ids'],   ids)
    interp.set_tensor(inp['input_mask'],  mask)
    interp.set_tensor(inp['segment_ids'], seg_ids)

    t0 = time.perf_counter()
    interp.invoke()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out = {t['name']: t['index'] for t in interp.get_output_details()}
    start_logits = interp.get_tensor(out['start_logits'])[0]
    end_logits   = interp.get_tensor(out['end_logits'])[0]

    answer, score = decode_answer(tokens, start_logits, end_logits, ctx_start)
    return answer, score, elapsed_ms

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MobileBERT question answering')
    parser.add_argument('--question', '-q', help='Question to ask')
    parser.add_argument('--context',  '-c', help='Passage of text to search')
    parser.add_argument('--file',     '-f', help='Read context from a text file')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--vocab', default=VOCAB)
    args = parser.parse_args()

    tokenizer     = WordpieceTokenizer(args.vocab)
    interp, device = load_model(args.model)
    print(f'Model loaded  ({device})', file=sys.stderr)

    # single-shot mode
    if args.question:
        if args.file:
            context = open(args.file).read().strip()
        elif args.context:
            context = args.context
        else:
            print('Provide --context or --file with --question', file=sys.stderr)
            sys.exit(1)
        answer, score, ms = ask(interp, tokenizer, args.question, context)
        print(f'Answer  : {answer}')
        print(f'Score   : {score:.2f}')
        print(f'Latency : {ms:.0f} ms')
        return

    # interactive mode
    print('Interactive mode — enter a passage, then ask questions.')
    print('Commands: :context  (change passage)   :quit  (exit)\n')
    context = None

    while True:
        if context is None:
            print('Paste your context passage (end with a blank line):')
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    sys.exit(0)
                if line == '':
                    break
                lines.append(line)
            context = ' '.join(lines).strip()
            if not context:
                continue
            print(f'\n[Context set — {len(context.split())} words]\n')

        try:
            question = input('Question: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not question:
            continue
        if question == ':quit':
            sys.exit(0)
        if question == ':context':
            context = None
            print()
            continue

        answer, score, ms = ask(interp, tokenizer, question, context)
        print(f'Answer  : {answer}')
        print(f'({ms:.0f} ms)\n')

if __name__ == '__main__':
    main()
